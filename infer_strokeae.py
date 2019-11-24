import os
import torch, numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from quickdraw.quickdraw import QuickDraw
from strokeae import RNNStrokeAE

def main( args ):
    qds = QuickDraw(args.root, categories=['face', 'airplane'], max_sketches_each_cat=2, mode=QuickDraw.STROKE,
        start_from_zero=True, verbose=True, seperate_p_tensor=True, shifted_seq_as_supevision=True)
    qdl = qds.get_dataloader(1)

    model = RNNStrokeAE(2, args.hidden, args.layers, 2, bidirectional=args.bidirec, ip_free_decoding=args.ip_free_dec)
    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()

    print(args.modelname)
    if os.path.exists(args.modelname + '.pth'):
        model.load_state_dict(torch.load(args.modelname + '.pth'))
    else:
        print('Model file not found, using random weights instead !')
    
    model.eval()
    
    with torch.no_grad():
        for i, ((X, _), (Y, _), _) in enumerate(qdl):
            h_initial = torch.zeros(args.layers * (2 if args.bidirec else 1), X.batch_sizes.max(), args.hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, h_initial = X.cuda(), h_initial.cuda()

            X_, l_ = pad_packed_sequence(X)
            if l_.item() < 3:
                continue
            Y_, _ = pad_packed_sequence(Y)
            if torch.cuda.is_available():
                X_numpy = X_.squeeze().cpu().numpy() # get rid of the obvious 1-th dimension which is 1 (because batch_size == 1)
                Y_numpy = Y_.squeeze().cpu().numpy()
            else:
                X_numpy = X_.squeeze().numpy()
                Y_numpy = Y_.squeeze().numpy()
            X_numpy = np.vstack((X_numpy, Y_numpy[-1,:])) # Append the last one; TODO: NEED FIXING

            latent = model.encoder(X, h_initial)
            if args.ip_free_dec:
                X_ = torch.zeros_like(X_) # Input free decoder
                X_ = pack_padded_sequence(X_, l_, enforce_sorted=False)
            out, P = model.decoder(X_, latent)
            if torch.cuda.is_available():
                out_numpy = out[0].cpu().numpy()
                P_numpy = P[0].cpu().numpy()
            else:
                out_numpy = out[0].numpy()
                P_numpy = P[0].cpu().numpy()
            out_numpy = np.vstack((np.array([0., 0.]), out_numpy))
            P_numpy = np.vstack((np.array([0.,]), P_numpy))

            fig, ax = plt.subplots(1, 2)
            ax[0].plot(X_numpy[:,0], X_numpy[:,1])
            ax[1].plot(out_numpy[:,0], out_numpy[:,1])
            plt.savefig(str(i) + '.png')
            plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--ip_free_dec', action='store_true', help='Decoder inputs are zero')
    parser.add_argument('--hidden', type=int, required=False, default=128, help='no. of hidden neurons')
    parser.add_argument('--layers', type=int, required=False, default=2, help='no of layers in RNN')
    parser.add_argument('--bidirec', action='store_true', help='Want the RNN to be bidirectional?')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    args = parser.parse_args()

    main( args )