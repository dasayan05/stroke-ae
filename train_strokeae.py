import sys, os
import torch, numpy as np
import torch.utils.tensorboard as tb
from torch.nn.utils.rnn import pad_packed_sequence

from quickdraw.quickdraw import QuickDraw
from strokeae import RNNStrokeAE, StrokeMSELoss

def main( args ):
    qds = QuickDraw(args.root, max_sketches_each_cat=200, mode=QuickDraw.STROKE, start_from_zero=True,
        verbose=True, problem=QuickDraw.ENCDEC)
    qdl = qds.get_dataloader(args.batch_size)

    # chosen device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = RNNStrokeAE(2, args.hidden, args.layers, 2, args.latent, bidirectional=True, ip_free_decoding=True,
        bezier_degree=args.bezier_degree, variational=args.variational)
    
    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    curloss = np.inf

    writer = tb.SummaryWriter(os.path.join(args.base, 'logs', args.tag))

    count = 0
    for e in range(args.epochs):

        model.train()
        for i, (X, (X_, Y, P), _) in enumerate(qdl):
            (Y, L), (P, _) = pad_packed_sequence(Y, batch_first=True), pad_packed_sequence(P, batch_first=True)
            
            strokemse = StrokeMSELoss(L.tolist(), bezier_degree=args.bezier_degree, bez_reg_weight=1e-2)
            l_optim = torch.optim.Adam(strokemse.parameters(), lr=args.lr)

            h_initial = torch.zeros(args.layers * 2, args.batch_size, args.hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, X_, Y, P, h_initial = X.cuda(), X_.cuda(), Y.cuda(), P.cuda(), h_initial.cuda()
            
            if args.anneal_KLD:
                # Annealing factor for KLD term
                # sigmoid = lambda x, T: 1. / (1. + np.exp(-x / T))
                linear = lambda e, e0: min(e / float(e0), 1.)
                anneal_factor = linear(e, 15)
            else:
                anneal_factor = 1.

            for _ in range(args.k_loptim):
                if args.variational:
                    (out, p), KLD = model(X, X_, h_initial)
                else:
                    out, p = model(X, X_, h_initial)

                KLD_loss = (KLD * args.latent * anneal_factor if args.variational else torch.tensor(0.))
                REC_loss = strokemse(out, p, Y, P, L) * (2 * args.hidden)

                loss = REC_loss + KLD_loss

                optim.zero_grad()
                l_optim.zero_grad()

                loss.backward()
                
                optim.step()
                l_optim.step()
            
            if i % args.interval == 0:
                count += 1
                if loss < curloss:
                    torch.save(model.state_dict(), os.path.join(args.base, args.modelname))
                    curloss = loss
                    saved = True
                else:
                    saved = False

                saved_string = ' (saved)' if saved else ''
                print(f'[Training: {i}/{e}/{args.epochs}] -> Loss: {REC_loss:.4f} + {KLD_loss:.4f}{saved_string}')
                
                writer.add_scalar('loss/recon', REC_loss.item(), global_step=count)
                writer.add_scalar('loss/anneal_factor', anneal_factor, global_step=count)
                writer.add_scalar('loss/KLD', KLD_loss.item(), global_step=count)
                writer.add_scalar('loss/total', loss.item(), global_step=count)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--base', type=str, required=False, default='.', help='base folder of operation (needed for condor)')

    parser.add_argument('-V', '--variational', action='store_true', help='Impose prior on latent space')
    parser.add_argument('--hidden', type=int, required=False, default=16, help='no. of hidden neurons')
    parser.add_argument('-l', '--latent', type=int, required=False, default=32, help='size of latent vector')
    parser.add_argument('--layers', type=int, required=False, default=1, help='no of layers in encoder RNN')
    parser.add_argument('-z', '--bezier_degree', type=int, required=False, default=5, help='degree of the bezier')
    
    parser.add_argument('-b','--batch_size', type=int, required=False, default=128, help='batch size')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=40, help='no of epochs')
    parser.add_argument('-k', '--k_loptim', type=int, required=False, default=4, help='k times optimize the local optimizer')
    parser.add_argument('--anneal_KLD', action='store_true', help='Increase annealing factor of KLD gradually')
    
    parser.add_argument('--tag', type=str, required=False, default='main', help='run identifier')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('-i', '--interval', type=int, required=False, default=100, help='logging interval')
    args = parser.parse_args()

    main( args )