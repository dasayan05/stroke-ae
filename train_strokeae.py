import sys, os, random
import torch, numpy as np
import torch.utils.tensorboard as tb
from torch.nn.utils.rnn import pad_packed_sequence

from quickdraw.quickdraw import QuickDraw
from strokeae import RNNStrokeAE, StrokeMSELoss
from infer_strokeae import inference

def length_gt(s, f):
    if len(s[0]) > f:
        return True, s
    else:
        return False, None

def main( args ):
    chosen_classes = [ 'cat', 'chair', 'face' , 'firetruck', 'mosquito', 'owl', 'pig', 'purse', 'shoe' ]

    qds = QuickDraw(args.root, categories=chosen_classes[:args.n_classes], raw=args.raw, max_sketches_each_cat=8000, mode=QuickDraw.STROKE, start_from_zero=True, verbose=True, problem=QuickDraw.ENCDEC)
    qdl = qds.get_dataloader(args.batch_size)
    
    qds_infer = QuickDraw(args.root, categories=chosen_classes[:args.n_classes], filter_func=lambda s: length_gt(s, 5),
        raw=args.raw, max_sketches_each_cat=15, mode=QuickDraw.STROKE, start_from_zero=True, verbose=True, problem=QuickDraw.ENCDEC)

    # chosen device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = RNNStrokeAE(2, args.hidden, args.layers, 2, args.latent, bidirectional=True,
        bezier_degree=args.bezier_degree, variational=args.variational)
    strokemse = StrokeMSELoss(args.bezier_degree, args.latent, bez_reg_weight_p=args.regp, bez_reg_weight_r=args.regr)
    
    model, strokemse = model.float(), strokemse.float()
    if torch.cuda.is_available():
        model = model.cuda()
        strokemse = strokemse.cuda()

    optim = torch.optim.Adam(list(model.parameters()) + list(strokemse.parameters()), lr=args.lr)

    writer = tb.SummaryWriter(os.path.join(args.base, 'logs', args.tag))

    count = 0
    for e in range(args.epochs):

        model.train()
        for i, (X, _) in enumerate(qdl):
            (Y, L) = pad_packed_sequence(X, batch_first=True)
            
            h_initial = torch.zeros(args.layers * 2, args.batch_size, args.hidden, dtype=torch.float32)
            c_initial = torch.zeros(args.layers * 2, args.batch_size, args.hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, Y, L, h_initial, c_initial = X.cuda(), Y.cuda(), L.cuda(), h_initial.cuda(), c_initial.cuda()
            
            if args.anneal_KLD:
                # Annealing factor for KLD term
                # sigmoid = lambda x, T: 1. / (1. + np.exp(-x / T))
                linear = lambda e, e0: min(e / float(e0), 1.)
                anneal_factor = linear(e, 20)
            else:
                anneal_factor = 1.

            if args.variational:
                latent, (out_ctrlpt, out_ratw), KLD = model(X, h_initial, c_initial)
            else:
                latent, (out_ctrlpt, out_ratw) = model(X, h_initial, c_initial)

            REC_loss = strokemse(out_ctrlpt, out_ratw, Y, L, latent)
            if args.variational:
                REC_loss = REC_loss * (2 * args.hidden)
                KLD_loss = KLD * anneal_factor * args.latent
            else:
                KLD_loss = torch.tensor(0.)

            loss = REC_loss + KLD_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if i % args.interval == 0:
                count += 1
                print(f'[Training: {i}/{e}/{args.epochs}] -> Loss: {REC_loss:.4f} + {KLD_loss:.4f}')
                
                writer.add_scalar('loss/recon', REC_loss.item(), global_step=count)
                writer.add_scalar('loss/anneal_factor', anneal_factor, global_step=count)
                writer.add_scalar('loss/KLD', KLD_loss.item(), global_step=count)
                writer.add_scalar('loss/total', loss.item(), global_step=count)

        # save after every epoch
        torch.save(model.state_dict(), os.path.join(args.base, args.modelname))

        model.eval()
        savefile = os.path.join(args.base, 'logs', args.tag, str(e) + '.png')
        inference(qds_infer.get_dataloader(1), model, layers=args.layers, hidden=args.hidden, variational=args.variational,
                bezier_degree=args.bezier_degree, savefile=savefile, nsamples=args.nsample, rsamples=args.rsample)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--base', type=str, required=False, default='.', help='base folder of operation (needed for condor)')
    parser.add_argument('--n_classes', '-c', type=int, required=False, default=3, help='no. of classes')
    parser.add_argument('--raw', action='store_true', help='Use raw QuickDraw data')

    parser.add_argument('-V', '--variational', action='store_true', help='Impose prior on latent space')
    parser.add_argument('--hidden', type=int, required=False, default=16, help='no. of hidden neurons')
    parser.add_argument('-l', '--latent', type=int, required=False, default=32, help='size of latent vector')
    parser.add_argument('--layers', type=int, required=False, default=1, help='no of layers in encoder RNN')
    parser.add_argument('-z', '--bezier_degree', type=int, required=False, default=5, help='degree of the bezier')
    
    parser.add_argument('-b','--batch_size', type=int, required=False, default=128, help='batch size')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=40, help='no of epochs')
    parser.add_argument('--anneal_KLD', action='store_true', help='Increase annealing factor of KLD gradually')
    parser.add_argument('--regp', type=float, required=False, default=1e-2, help='Regularizer weight on control points')
    parser.add_argument('--regr', type=float, required=False, default=1e-5, help='Regularizer weight on rational weights')
    
    parser.add_argument('--tag', type=str, required=False, default='main', help='run identifier')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('-i', '--interval', type=int, required=False, default=100, help='logging interval')
    parser.add_argument('--nsample', type=int, required=False, default=6, help='no. of data samples for inference')
    parser.add_argument('--rsample', type=int, required=False, default=6, help='no. of distribution samples for inference')
    args = parser.parse_args()

    main( args )