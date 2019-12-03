import os
import torch, numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from quickdraw.quickdraw import QuickDraw
from strokeae import RNNStrokeAE
from beziercurve import draw_bezier

def main( args ):
    qds = QuickDraw(args.root, categories=['face', 'airplane'], max_sketches_each_cat=2, mode=QuickDraw.STROKE,
        start_from_zero=True, verbose=True, problem=QuickDraw.ENCDEC)
    qdl = qds.get_dataloader(1)

    model = RNNStrokeAE(2, args.hidden, args.layers, 2, args.latent, bidirectional=True, ip_free_decoding=True,
        bezier_degree=args.bezier_degree, variational=args.vae)
    
    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()

    # print(args.modelname)
    if os.path.exists(args.modelname):
        model.load_state_dict(torch.load(args.modelname))
    else:
        raise 'Model file not found !'
    
    model.eval()
    
    with torch.no_grad():
        for i, (X, (_, _, _), _) in enumerate(qdl):
            h_initial = torch.zeros(args.layers * 2, 1, args.hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, h_initial = X.cuda(), h_initial.cuda()

            X_, l_ = pad_packed_sequence(X)
            if l_.item() < 3:
                continue

            if torch.cuda.is_available():
                X_numpy = X_.squeeze().cpu().numpy() # get rid of the obvious 1-th dimension which is 1 (because batch_size == 1)
            else:
                X_numpy = X_.squeeze().numpy()

            h_init = model.encoder(X, h_initial)

            curve = np.empty((0, 2))
            x_init = torch.tensor([[0., 0.]], dtype=torch.float32)
            px = pack_padded_sequence(x_init.unsqueeze(1), torch.tensor([1]), enforce_sorted=False)
            
            if torch.cuda.is_available():
                px = px.cuda()
            
            model.eval()
            stop = False
            with torch.no_grad():
                while not stop:
                    (y, p), s = model.decoder(px, h_init, return_state=True)
                    curve = np.vstack((curve, y[0].detach().cpu().numpy()))
                    if curve.shape[0] >= (args.bezier_degree + 1):
                        break
                    if not args.ip_free_dec:
                        px = pack_padded_sequence(y[0].unsqueeze(1), torch.tensor([1]), enforce_sorted=False)
                    h_init = s
                    if args.bezier_degree == 0:
                        stop = True if p[0].item() > 0.75 else False

            fig, ax = plt.subplots(1, 2)
            ax[0].plot(X_numpy[:,0], X_numpy[:,1])
            if args.bezier_degree != 0:
                print(curve)
                draw_bezier(curve, annotate=False, draw_axis=ax[1])
            else:
                ax[1].plot(curve[:,0], curve[:,1])
            plt.savefig(str(i) + '.png')
            plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--hidden', type=int, required=False, default=128, help='no. of hidden neurons')
    parser.add_argument('-l', '--latent', type=int, required=False, default=32, help='size of latent vector')
    parser.add_argument('--ip_free_dec', action='store_true', help='Decoder inputs are zero')
    parser.add_argument('--layers', type=int, required=False, default=2, help='no of layers in RNN')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('-z', '--bezier_degree', type=int, required=False, default=0, help='degree of the bezier')
    parser.add_argument('-V', '--vae', action='store_true', help='Impose prior on latent space')
    args = parser.parse_args()

    main( args )