import sys, os
import torch, numpy as np
import torch.utils.tensorboard as tb
from torch.nn.utils.rnn import pad_packed_sequence

# from quickdrawstrokes import QuickDrawStrokes
from quickdraw.quickdraw import QuickDraw
from strokeae import RNNStrokeAE, StrokeMSELoss

def main( args ):
    qds = QuickDraw(args.root, max_sketches_each_cat=200, mode=QuickDraw.STROKE, start_from_zero=True,
        verbose=True, problem=QuickDraw.ENCDEC)
    qds_train, qds_test = qds.split(0.8)
    qdl_train = qds_train.get_dataloader(args.batch_size)
    qdl_test = qds_test.get_dataloader(args.batch_size)

    model = RNNStrokeAE(2, args.hidden, args.layers, 2, bidirectional=args.bidirec, ip_free_decoding=args.ip_free_dec,
        bezier_degree=args.bezier_degree, variational=args.vae)
    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    curloss = np.inf

    writer = tb.SummaryWriter(os.path.join('.', 'logs', args.tag))

    count = 0
    for e in range(args.epochs):

        model.train()
        for i, (X, (X_, Y, P), _) in enumerate(qdl_train):
            (Y, L), (P, _) = pad_packed_sequence(Y, batch_first=True), pad_packed_sequence(P, batch_first=True)
            strokemse = StrokeMSELoss(L.tolist(), bezier_degree=args.bezier_degree)
            l_optim = torch.optim.Adam(strokemse.parameters(), lr=args.lr)

            h_initial = torch.zeros(args.layers * (2 if args.bidirec else 1), X.batch_sizes.max(), args.hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, X_, Y, P = X.cuda(), X_.cuda(), Y.cuda(), P.cuda()
                h_initial = h_initial.cuda()
            
            if args.anneal_k:
                # Reduce K from 'args.k_loptim' to 1 step-by-step
                # after every (args.epochs // args.k_loptim) epochs
                K = args.k_loptim - (e // (args.epochs // args.k_loptim))
            else:
                K = args.k_loptim

            if args.anneal_KLD:
                # Annealing factor for KLD term
                anneal_factor = 1. - np.exp(- (e + 1) / (args.max_KLD_epoch / 4.6))
            else:
                anneal_factor = 1.

            for _ in range(K):
                if args.vae:
                    (out, p), KLD = model(X, X_, h_initial)
                else:
                    out, p = model(X, X_, h_initial)

                KLD_loss = (KLD * anneal_factor if args.vae else 0)
                REC_loss = strokemse(out, p, Y, P, L)

                loss = REC_loss + KLD_loss

                optim.zero_grad()
                l_optim.zero_grad()

                loss.backward()
                
                optim.step()
                l_optim.step()
            
            if i % args.interval == 0:
                count += 1
                if loss < curloss:
                    torch.save(model.state_dict(), args.modelname + '.pth')
                    curloss = loss
                    saved = True
                else:
                    saved = False

                saved_string = ' (saved)' if saved else ''
                print(f'[Training: {i}/{e}/{args.epochs}] -> Loss: {REC_loss:.4f} + {KLD_loss:.4f}{saved_string}')
                
                writer.add_scalar('loss/total', loss.item(), global_step=count)
                writer.add_scalar('loss/recon', REC_loss.item(), global_step=count)
                writer.add_scalar('loss/KLD', KLD_loss.item(), global_step=count)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--ip_free_dec', action='store_true', help='Decoder inputs are zero')
    parser.add_argument('-b','--batch_size', type=int, required=False, default=32, help='batch size')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('--hidden', type=int, required=False, default=128, help='no. of hidden neurons')
    parser.add_argument('--layers', type=int, required=False, default=2, help='no of layers in encoder RNN')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=500, help='no of epochs')
    parser.add_argument('--dropout', type=float, required=False, default=0.5, help='dropout fraction')
    parser.add_argument('--bidirec', action='store_true', help='Want the encoder RNN to be bidirectional?')
    parser.add_argument('-i', '--interval', type=int, required=False, default=100, help='logging interval')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('--tag', type=str, required=False, default='main', help='run identifier')
    parser.add_argument('-z', '--bezier_degree', type=int, required=False, default=0, help='degree of the bezier')
    parser.add_argument('-k', '--k_loptim', type=int, required=False, default=4, help='k times optimize the local optimizer')
    parser.add_argument('--anneal_k', action='store_true', help='Decrease K gradually')
    parser.add_argument('-V', '--vae', action='store_true', help='Impose prior on latent space')
    parser.add_argument('--anneal_KLD', action='store_true', help='Increase annealing factor of KLD gradually')
    parser.add_argument('--max_KLD_epoch', type=int, required=False, default=5000, help='Saturate KLD anneal_factor to 1 at this epoch')
    args = parser.parse_args()

    main( args )