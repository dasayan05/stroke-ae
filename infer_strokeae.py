import os
import torch, numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from quickdraw.quickdraw import QuickDraw
from strokeae import RNNStrokeAE
from beziercurve import draw_bezier

def decode_stroke(decoder, latent, bezier_degree, dtype=torch.float32):
    curve = np.empty((0, 2))
    x_init = torch.tensor([[0., 0.]], dtype=dtype)
    px = pack_padded_sequence(x_init.unsqueeze(1), torch.tensor([1]), enforce_sorted=False)
    
    if torch.cuda.is_available():
        px = px.cuda()
    
    with torch.no_grad():
        while True:
            y, s = decoder(px, latent, return_state=True)
            next_pt = y[0].detach().cpu().numpy()
            curve = np.vstack((curve, next_pt))
            if curve.shape[0] >= (bezier_degree + 1):
                break
            px = pack_padded_sequence(y[0].unsqueeze(1), torch.tensor([1]), enforce_sorted=False)
            latent = s

    return curve

def inference(qdl, model, layers, hidden, nsamples, rsamples, variational, bezier_degree, savefile):
    with torch.no_grad():
        fig, ax = plt.subplots(nsamples, (rsamples + 1), figsize=(rsamples * 4, nsamples * 4))
        for i, (X, (_, _, _), _) in enumerate(qdl):
            if i >= nsamples:
                break

            h_initial = torch.zeros(layers * 2, 1, hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, h_initial = X.cuda(), h_initial.cuda()

            X_, l_ = pad_packed_sequence(X)

            if torch.cuda.is_available():
                X_numpy = X_.squeeze().cpu().numpy() # get rid of the obvious 1-th dimension which is 1 (because batch_size == 1)
            else:
                X_numpy = X_.squeeze().numpy()

            if not variational:
                latent = model.encoder(X, h_initial)
                normal = torch.distributions.Normal(latent.squeeze(), torch.zeros_like(latent.squeeze()))
            else:
                mu, sigma = model.encoder(X, h_initial)
                normal = torch.distributions.Normal(mu.squeeze(), sigma.squeeze())

            ax[i, 0].scatter(X_numpy[:, 0], X_numpy[:,1])
            ax[i, 0].plot(X_numpy[:,0], X_numpy[:,1])

            for s in range(rsamples):
                latent = normal.sample().unsqueeze(0)
                curve = decode_stroke(model.decoder, latent, bezier_degree=bezier_degree)

                if bezier_degree != 0:
                    draw_bezier(curve, annotate=False, draw_axis=ax[i, s + 1])
                else:
                    ax[i, s + 1].plot(curve[:,0], curve[:,1])
            
        plt.savefig(savefile)
        plt.xticks([]); plt.yticks([])
        plt.close()

def main( args ):
    qds = QuickDraw(args.root, categories=['face', 'airplane'], max_sketches_each_cat=2, mode=QuickDraw.STROKE,
        start_from_zero=True, verbose=True, problem=QuickDraw.ENCDEC)
    qdl = qds.get_dataloader(1)

    model = RNNStrokeAE(2, args.hidden, args.layers, 2, args.latent, bidirectional=True,
        bezier_degree=args.bezier_degree, variational=args.variational)

    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()

    if os.path.exists(args.modelname):
        model.load_state_dict(torch.load(args.modelname))
    else:
        raise 'Model file not found !'

    model.eval()
    inference(qdl, model, nsamples=args.nsamples, rsamples=args.rsamples, variational=args.variational,
        hidden=args.hidden, layers=args.layers, bezier_degree=args.bezier_degree, savefile=args.savefile)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--hidden', type=int, required=False, default=128, help='no. of hidden neurons')
    parser.add_argument('-l', '--latent', type=int, required=False, default=32, help='size of latent vector')
    parser.add_argument('--layers', type=int, required=False, default=2, help='no of layers in RNN')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('-z', '--bezier_degree', type=int, required=False, default=0, help='degree of the bezier')
    parser.add_argument('-V', '--variational', action='store_true', help='Impose prior on latent space')
    parser.add_argument('-n', '--nsamples', type=int, required=False, default=5, help='no. of data samples')
    parser.add_argument('-s', '--rsamples', type=int, required=False, default=4, help='no. of samples drawn from p(z|x); only if VAE')
    parser.add_argument('--savefile', type=str, required=True, help='saved filename')
    args = parser.parse_args()

    main( args )