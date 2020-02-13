import os
import torch, numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence

from quickdraw.quickdraw import QuickDraw
from beziercurve import draw_bezier

def inference(qdl, model, layers, hidden, nsamples, bezier_degree_low, bezier_degree_high, savefile):
    with torch.no_grad():
        rsamples = bezier_degree_high - bezier_degree_low + 1
        fig, ax = plt.subplots(nsamples, (rsamples + 1), figsize=((rsamples + 1) * 4, nsamples * 4))
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

            if model.rational:
                ctrlpt, ratw = model(X, h_initial, c_initial)
            else:
                ctrlpt = model(X, h_initial, c_initial)

            # normal = torch.distributions.Normal(ctrlpt.squeeze(), torch.zeros_like(ctrlpt.squeeze()))

            ax[i, 0].scatter(X_numpy[:, 0], X_numpy[:,1])
            ax[i, 0].plot(X_numpy[:,0], X_numpy[:,1])
            ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])

            for z in range(bezier_degree_low, bezier_degree_high + 1):
                ctrlpt_ = ctrlpt[z - bezier_degree_low].squeeze()
                
                if model.rational:
                    ratw_ = ratw[z - bezier_degree_low].squeeze()
                    ratw_ = torch.cat([torch.tensor([5.,], device=ratw_.device), ratw_, torch.tensor([5.,], device=ratw_.device)], 0)
                    ratw_ = torch.sigmoid(ratw_)
                else:
                    ratw_ = None

                # Decode the encoded DelP1..DelPn
                P0 = torch.zeros(1, ctrlpt_.shape[1], device=ctrlpt_.device)
                ctrlpt_ = torch.cat([P0, ctrlpt_], 0)
                ctrlpt_ = torch.cumsum(ctrlpt_, 0)
                draw_bezier(ctrlpt_.cpu().numpy(), ratw_.cpu().numpy() if ratw_ is not None else ratw_,
                    annotate=False, draw_axis=ax[i, z - bezier_degree_low + 1])
                ax[i, z - bezier_degree_low + 1].set_xticks([])
                ax[i, z - bezier_degree_low + 1].set_yticks([])
            
        plt.savefig(savefile)
        plt.close()