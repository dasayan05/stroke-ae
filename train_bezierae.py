import sys, os, random
import torch, numpy as np
import torch.utils.tensorboard as tb
from torch.nn.utils.rnn import pad_packed_sequence

from quickdraw.quickdraw import QuickDraw
from bezierae import RNNBezierAE
from infer_bezierae import inference

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

    model = RNNBezierAE(2, args.hidden, args.layers, args.bezier_degree, bidirectional=True,
        variational=args.variational, dropout=args.dropout)
    
    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()

    mseloss = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.8)

    writer = tb.SummaryWriter(os.path.join(args.base, 'logs', args.tag))

    count = 0
    for e in range(args.epochs):

        model.train()
        for i, (X, _) in enumerate(qdl):
            # break
            h_initial = torch.zeros(args.layers * 2, args.batch_size, args.hidden, dtype=torch.float32)
            c_initial = torch.zeros(args.layers * 2, args.batch_size, args.hidden, dtype=torch.float32)
            
            if torch.cuda.is_available():
                X, h_initial, c_initial = X.cuda(), h_initial.cuda(), c_initial.cuda()

            # Unpacking the X, nothing more
            X_, L_ = pad_packed_sequence(X, batch_first=True)
            
            if args.anneal_KLD:
                # Annealing factor for KLD term
                linear = lambda e, e0: min(e / float(e0), 1.)
                anneal_factor = linear(e, 15)
            else:
                anneal_factor = 1.

            if args.variational:
                out, regu, KLD = model(X, h_initial, c_initial)
            else:
                out, regu = model(X, h_initial, c_initial)


            batch_losses = []
            for o, x_, l_ in zip(out, X_, L_):
                # per sample iteration
                batch_losses.append( mseloss(o[:l_, :], x_[:l_, :]) )
            
            REC_loss = sum(batch_losses) / len(batch_losses)
            REC_loss = REC_loss + regu * args.regp

            if args.variational:
                KLD_loss = KLD * (args.bezier_degree * 2) * anneal_factor
            else:
                KLD_loss = torch.tensor(0.)

            loss = REC_loss + KLD_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if i % args.interval == 0:
                count += 1
                print(f'[Training: {i}/{e}/{args.epochs}] -> Loss: {loss:.6f}')
                
                if args.variational:
                    writer.add_scalar('train/loss/REC', REC_loss.item(), global_step=count)
                    writer.add_scalar('train/loss/KLD', KLD_loss.item(), global_step=count)
                    writer.add_scalar('train/loss/total', loss.item(), global_step=count)
                else:
                    writer.add_scalar('train/loss/total', loss.item(), global_step=count)

        # save after every epoch
        torch.save(model.state_dict(), os.path.join(args.base, args.modelname))

        model.eval()
        savefile = os.path.join(args.base, 'logs', args.tag, str(e) + '.png')
        inference(qds_infer.get_dataloader(1), model, layers=args.layers, hidden=args.hidden, variational=args.variational,
                bezier_degree=args.bezier_degree, savefile=savefile, nsamples=args.nsample, rsamples=args.rsample)

        # invoke scheduler
        sched.step()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--base', type=str, required=False, default='.', help='base folder of operation (needed for condor)')
    parser.add_argument('--n_classes', '-c', type=int, required=False, default=3, help='no. of classes')
    parser.add_argument('--raw', action='store_true', help='Use raw QuickDraw data')

    parser.add_argument('-V', '--variational', action='store_true', help='Impose prior on latent space')
    parser.add_argument('--hidden', type=int, required=False, default=16, help='no. of hidden neurons')
    parser.add_argument('--layers', type=int, required=False, default=1, help='no of layers in encoder RNN')
    parser.add_argument('-z', '--bezier_degree', type=int, required=False, default=5, help='degree of the bezier')
    
    parser.add_argument('-b','--batch_size', type=int, required=False, default=128, help='batch size')
    parser.add_argument('--dropout', type=float, required=False, default=0.8, help='Dropout rate')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=40, help='no of epochs')
    parser.add_argument('--anneal_KLD', action='store_true', help='Increase annealing factor of KLD gradually')
    parser.add_argument('--regp', type=float, required=False, default=1e-2, help='Regularizer weight on control points')
    
    parser.add_argument('--tag', type=str, required=False, default='main', help='run identifier')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('-i', '--interval', type=int, required=False, default=100, help='logging interval')
    parser.add_argument('--nsample', type=int, required=False, default=6, help='no. of data samples for inference')
    parser.add_argument('--rsample', type=int, required=False, default=6, help='no. of distribution samples for inference')
    args = parser.parse_args()

    main( args )