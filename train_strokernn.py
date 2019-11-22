import os, pdb
import torch, numpy as np
from torch.nn.utils.rnn import pad_packed_sequence

from quickdraw.quickdraw import QuickDraw
from strokernn import StrokeRNN
from strokeae import StrokeMSELoss

def main( args ):
    dataset = QuickDraw(args.root, categories=['face'], max_samples=75000, mode=QuickDraw.SKETCH, verbose=True,
        seperate_p_tensor=True, shifted_seq_as_supevision=True,
        filter_func=lambda s: (True, [s[0],]))
    dataloader = dataset.get_dataloader(batch_size=args.batch_size)

    model = StrokeRNN(n_input=2, n_hidden=args.hidden, n_layer=args.layers, n_output=2, dropout=args.dropout)
    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    strokemse = StrokeMSELoss()

    curloss = np.inf
    
    for e in range(args.epochs):
        for i, ((X, _), (Y, P), _) in enumerate(dataloader):

            (Y, _), (P, _) = pad_packed_sequence(Y, batch_first=True), pad_packed_sequence(P, batch_first=True)
            h_initial = torch.zeros(args.layers, X.batch_sizes.max(), args.hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, Y, P = X.cuda(), Y.cuda(), P.cuda()
                h_initial = h_initial.cuda()

            (out, p), L = model(X, h_initial)
            
            # breakpoint()
            loss = strokemse(out, p, Y, P, L)

            if i % 100 == 0:
                if loss < curloss:
                    torch.save(model.state_dict(), 'model.pth')
                    curloss = loss
                    saved = True
                else:
                    saved = False

                saved_string = ' (saved)' if saved else ''
                print(f'[{i}/{e}/{args.epochs}] -> Loss: {loss}{saved_string}')
            
            optim.zero_grad()
            loss.backward()
            optim.step()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='QuickDraw dataset root (location of .bin files)')
    parser.add_argument('-b','--batch_size', type=int, required=False, default=32, help='batch size')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('--hidden', type=int, required=False, default=128, help='no. of hidden neurons')
    parser.add_argument('--layers', type=int, required=False, default=2, help='no of layers in RNN')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=500, help='no of epochs')
    parser.add_argument('--dropout', type=float, required=False, default=0.5, help='dropout fraction')
    args = parser.parse_args()

    main( args )