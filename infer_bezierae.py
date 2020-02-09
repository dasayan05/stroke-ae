import os
import torch, numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence

from quickdraw.quickdraw import QuickDraw
from beziercurve import draw_bezier

def inference(qdl, model, layers, hidden, nsamples, rsamples, variational, bezier_degree, savefile):
    with torch.no_grad():
        fig, ax = plt.subplots(nsamples, (rsamples + 1), figsize=(rsamples * 4, nsamples * 4))
        for i, (X, _) in enumerate(qdl):
            if i >= nsamples:
                break

            h_initial = torch.zeros(layers * 2, 1, hidden, dtype=torch.float32)
            c_initial = torch.zeros(layers * 2, 1, hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, h_initial, c_initial = X.cuda(), h_initial.cuda(), c_initial.cuda()

            X_, l_ = pad_packed_sequence(X)

            if torch.cuda.is_available():
                X_numpy = X_.squeeze().cpu().numpy() # get rid of the obvious 1-th dimension which is 1 (because batch_size == 1)
            else:
                X_numpy = X_.squeeze().numpy()

            if not variational:
                if model.rational:
                    ctrlpt, ratw = model(X, h_initial, c_initial)
                else:
                    ctrlpt = model(X, h_initial, c_initial)
                normal = torch.distributions.Normal(ctrlpt.squeeze(), torch.zeros_like(ctrlpt.squeeze()))
            else:
                ctrlpt_mu, ctrlpt_std, ratw = model(X, h_initial, c_initial)
                ctrlpt_mu, ctrlpt_std = ctrlpt_mu.view(-1, bezier_degree + 1, 2), ctrlpt_std.view(-1, bezier_degree + 1, 2)
                normal = torch.distributions.Normal(ctrlpt_mu.squeeze(), ctrlpt_std.squeeze())

            ax[i, 0].scatter(X_numpy[:, 0], X_numpy[:,1])
            ax[i, 0].plot(X_numpy[:,0], X_numpy[:,1])

            for s in range(rsamples):
                ctrlpt_ = normal.sample()
                
                if model.rational:
                    ratw_ = ratw.squeeze()
                    ratw_ = torch.cat([torch.tensor([5.,], device=ratw_.device), ratw_, torch.tensor([5.,], device=ratw_.device)], 0)
                    ratw_ = torch.sigmoid(ratw_)
                else:
                    ratw_ = None

                # Decode the encoded DelP1..DelPn
                P0 = torch.zeros(1, ctrlpt_.shape[1], device=ctrlpt_.device)
                ctrlpt_ = torch.cat([P0, ctrlpt_], 0)
                ctrlpt_ = torch.cumsum(ctrlpt_, 0)
                draw_bezier(ctrlpt_.cpu().numpy(), ratw_.cpu().numpy() if ratw_ is not None else ratw_,
                    annotate=False, draw_axis=ax[i, s + 1])
            
        plt.xticks([]); plt.yticks([])
        plt.savefig(savefile)
        plt.close()