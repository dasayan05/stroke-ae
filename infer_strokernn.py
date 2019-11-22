import torch
import pdb, numpy as np
import matplotlib.pyplot as plt
plt.ion()

from strokernn import StrokeRNN
from torch.nn.utils.rnn import pack_padded_sequence

def main( args ):
    model = StrokeRNN(n_input=2, n_hidden=args.hidden, n_layer=args.layers, n_output=2, dropout=args.dropout)
    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('model.pth'))

    while True:
        curve = np.empty((0, 2))

        x_init_np = np.random.multivariate_normal([0.,0.], np.eye(2) * 0.01)
        # x_init_np = np.zeros_like(x_init_np)
        x_init = torch.tensor(x_init_np[np.newaxis, :], dtype=torch.float32)

        h_init = torch.rand(args.layers, 1, args.hidden, dtype=torch.float32)
        # h_init = torch.zeros_like(h_init)
        px = pack_padded_sequence(x_init.unsqueeze(1), torch.tensor([1]), enforce_sorted=False)
        
        if torch.cuda.is_available():
            px, h_init = px.cuda(), h_init.cuda()
        
        model.eval()
        stop = False
        with torch.no_grad():
            while not stop:
                print('gen')
                (y, p), s = model(px, h_init)
                # print(y[0])
                curve = np.vstack((curve, y[0].detach().cpu().numpy()))
                px = pack_padded_sequence(y[0].unsqueeze(1), torch.tensor([1]), enforce_sorted=False)
                h_init = s
                print(p[0].item())
                stop = True if p[0].item() > 0.5 else False

        plt.plot(curve[:,0], curve[:,1])
        print('Curve produced')
        plt.pause(1)
        plt.clf()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, required=False, default=128, help='no. of hidden neurons')
    parser.add_argument('--layers', type=int, required=False, default=2, help='no of layers in RNN')
    parser.add_argument('--dropout', type=float, required=False, default=0.5, help='dropout fraction')
    args = parser.parse_args()

    main( args )