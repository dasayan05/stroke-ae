import os, random
import torch, numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from quickdraw.quickdraw import QuickDraw
from strokeae import RNNBezierAE

def main( args ):
    chosen_classes = [q.split('.')[0] for q in os.listdir(args.root)]
    random.shuffle(chosen_classes)
    qds = QuickDraw(args.root, categories=chosen_classes[:2], max_sketches_each_cat=2, mode=QuickDraw.STROKE,
        start_from_zero=True, verbose=True, problem=QuickDraw.ENCDEC)
    qdl = qds.get_dataloader(1)

    model = RNNBezierAE(2, args.hidden, args.layers, args.latent, bidirectional=True, variational=args.variational)
    
    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()

    if os.path.exists(args.modelname):
        model.load_state_dict(torch.load(args.modelname))
    else:
        raise 'Model file not found !'
    
    model.eval()
    
    with torch.no_grad():
        fig, ax = plt.subplots(args.nsamples, 2 if not args.variational else (args.rsamples + 1),
            figsize=(8 if not args.variational else args.rsamples * 4, args.nsamples * 4))
        for i, (X, (_, _, _), _) in enumerate(qdl):
            if i >= args.nsamples:
                break
            
            h_initial = torch.zeros(args.layers * 2, 1, args.hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, h_initial = X.cuda(), h_initial.cuda()

            X_, l_ = pad_packed_sequence(X)

            if torch.cuda.is_available():
                X_numpy = X_.squeeze().cpu().numpy() # get rid of the obvious 1-th dimension which is 1 (because batch_size == 1)
            else:
                X_numpy = X_.squeeze().numpy()

            if not args.variational:
                latent = model.encoder(X, h_initial)
                normal = torch.distributions.Normal(latent.squeeze(), torch.zeros_like(latent.squeeze()))
            else:
                mu, sigma = model.encoder(X, h_initial)
                normal = torch.distributions.Normal(mu.squeeze(), sigma.squeeze())

            t = torch.arange(0.0, 1.0 + args.granularity, args.granularity).unsqueeze(0).unsqueeze(-1)
            if torch.cuda.is_available():
                t = t.cuda()

            ax[i, 0].scatter(X_numpy[:, 0], X_numpy[:,1])
            ax[i, 0].plot(X_numpy[:,0], X_numpy[:,1])

            for s in range(1 if not args.variational else args.rsamples):
                latent = normal.sample().unsqueeze(0)
                out = model.decoder(t, latent)
                out = out.squeeze().cpu().numpy()

                ax[i, s + 1].scatter(out[:, 0], out[:,1])
                ax[i, s + 1].plot(out[:,0], out[:,1])
                ax[i, s + 1].get_xaxis().set_visible(False)
                ax[i, s + 1].get_yaxis().set_visible(False)

        plt.savefig(args.savefile)
        plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--hidden', type=int, required=False, default=128, help='no. of hidden neurons')
    parser.add_argument('-l', '--latent', type=int, required=False, default=32, help='size of latent vector')
    parser.add_argument('--layers', type=int, required=False, default=2, help='no of layers in RNN')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('-V', '--variational', action='store_true', help='Impose prior on latent space')
    parser.add_argument('-g', '--granularity', type=float, required=False, default=1./20., help='granularity of the curve')
    parser.add_argument('-n', '--nsamples', type=int, required=False, default=5, help='no. of data samples')
    parser.add_argument('-s', '--rsamples', type=int, required=False, default=4, help='no. of samples drawn from p(z|x); only if VAE')
    parser.add_argument('--savefile', type=str, required=True, help='saved filename')
    args = parser.parse_args()

    main( args )