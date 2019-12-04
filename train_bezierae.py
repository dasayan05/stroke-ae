import sys, os, random
import torch, numpy as np
import torch.utils.tensorboard as tb
from torch.nn.utils.rnn import pad_packed_sequence

from quickdraw.quickdraw import QuickDraw
from strokeae import RNNBezierAE, StrokeMSELoss

def main( args ):
    chosen_classes = [q.split('.')[0] for q in os.listdir(args.root)]
    random.shuffle(chosen_classes)
    qds = QuickDraw(args.root, categories=chosen_classes[:25], max_sketches_each_cat=2000, mode=QuickDraw.STROKE,
        start_from_zero=True, verbose=True, problem=QuickDraw.ENCDEC)
    qdl = qds.get_dataloader(args.batch_size)

    # chosen device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = RNNBezierAE(2, args.hidden, args.layers, args.latent, bidirectional=True, variational=args.variational)
    
    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()

    mseloss = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    curloss = np.inf

    writer = tb.SummaryWriter(os.path.join(args.base, 'logs', args.tag))

    count = 0
    for e in range(args.epochs):

        model.train()
        for i, (X, (_, _, _), _) in enumerate(qdl):
            h_initial = torch.zeros(args.layers * 2, args.batch_size, args.hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, h_initial = X.cuda(), h_initial.cuda()

            # Unpacking the X, nothing more
            X_, L_ = pad_packed_sequence(X, batch_first=True)
            
            if args.anneal_KLD:
                # Annealing factor for KLD term
                linear = lambda e, e0: min(e / float(e0), 1.)
                anneal_factor = linear(e, 15)
            else:
                anneal_factor = 1.

            if args.variational:
                out, KLD = model(X, h_initial)
            else:
                out = model(X, h_initial)

            batch_losses = []
            for o, x_, l_ in zip(out, X_, L_):
                # per sample iteration
                batch_losses.append( mseloss(o[:l_, :], x_[:l_, :]) )
            
            if args.variational:
                KLD_loss = KLD * args.latent * anneal_factor
                REC_loss = sum(batch_losses) / len(batch_losses) * (2 * args.hidden)
            else:
                REC_loss = sum(batch_losses) / len(batch_losses)
                KLD_loss = torch.tensor(0.)

            loss = REC_loss + KLD_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if i % args.interval == 0:
                count += 1
                if loss < curloss:
                    torch.save(model.state_dict(), os.path.join(args.base, args.modelname))
                    curloss = loss
                    saved = True
                else:
                    saved = False

                saved_string = ' (saved)' if saved else ''
                print(f'[Training: {i}/{e}/{args.epochs}] -> Loss: {loss:.6f}{saved_string}')
                
                if args.variational:
                    writer.add_scalar('train/loss/REC', REC_loss.item(), global_step=count)
                    writer.add_scalar('train/loss/KLD', KLD_loss.item(), global_step=count)
                    writer.add_scalar('train/loss/total', loss.item(), global_step=count)
                else:
                    writer.add_scalar('train/loss/total', loss.item(), global_step=count)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--base', type=str, required=False, default='.', help='base folder of operation (needed for condor)')

    parser.add_argument('-V', '--variational', action='store_true', help='Impose prior on latent space')
    parser.add_argument('--hidden', type=int, required=False, default=16, help='no. of hidden neurons')
    parser.add_argument('-l', '--latent', type=int, required=False, default=32, help='size of latent vector')
    parser.add_argument('--layers', type=int, required=False, default=1, help='no of layers in encoder RNN')
    
    parser.add_argument('-b','--batch_size', type=int, required=False, default=128, help='batch size')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=40, help='no of epochs')
    parser.add_argument('--anneal_KLD', action='store_true', help='Increase annealing factor of KLD gradually')
    
    parser.add_argument('--tag', type=str, required=False, default='main', help='run identifier')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('-i', '--interval', type=int, required=False, default=100, help='logging interval')
    args = parser.parse_args()

    main( args )