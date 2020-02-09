import os
import torch, numpy as np
dist = torch.distributions
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from quickdraw.quickdraw import QuickDraw
from beziercurve import draw_bezier

def drawsketch(ctrlpts, ratws, st_starts, n_stroke, draw_axis=plt.gca()):
    ctrlpts, ratws, st_starts = ctrlpts[:n_stroke], ratws[:n_stroke], st_starts[:n_stroke]
    ctrlpts = ctrlpts.view(-1, ctrlpts.shape[-1] // 2, 2)
    
    z_ = torch.ones((ratws.shape[0], 1), device=ratws.device) * 5. # sigmoid(5.) is close to 1
    ratws = torch.cat([z_, ratws, z_], 1)
    for ctrlpt, ratw, st_start in zip(ctrlpts, torch.sigmoid(ratws), st_starts):
        ctrlpt = ctrlpt.detach().cpu().numpy()
        ratw = ratw.detach().cpu().numpy()
        st_start = st_start.detach().cpu().numpy()
        draw_bezier(ctrlpt, ratw, start_xy=st_start, draw_axis=draw_axis, annotate=False,
            ctrlPointPlotKwargs=dict(marker='X', color='red', linestyle='--', alpha=0.4))
    draw_axis.invert_yaxis()

def stroke_embed(batch, initials, embedder, variational=False):
    h_initial, c_initial = initials
    # Redundant, but thats fine
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # accumulate all info into these empty lists
    sketches_ctrlpt, sketches_ratw, sketches_st_starts, sketches_stopbits = [], [], [], []
    n_strokes = []
    
    for sk, _ in batch:
        # for each sketch in the batch
        st_starts = torch.tensor([st[0,:2] for st in sk], device=device)
        sk = [torch.tensor(st[:,:-1], device=device) - st_start for st, st_start in zip(sk, st_starts)]
        ls = [st.shape[0] for st in sk]
        sk = pad_sequence(sk, batch_first=True)
        sk = pack_padded_sequence(sk, ls, batch_first=True, enforce_sorted=False)
        if not variational:
            if embedder.rational:
                emb_ctrlpt, emb_ratw = embedder(sk, h_initial, c_initial)
            else:
                emb_ctrlpt = embedder(sk, h_initial, c_initial)
        else:
            if embedder.rational:
                emb_ctrlpt, _, emb_ratw = embedder(sk, h_initial, c_initial)
            else:
                emb_ctrlpt, _ = embedder(sk, h_initial, c_initial)
        
        sketches_ctrlpt.append(emb_ctrlpt.view(len(ls), -1))
        if embedder.rational:
            sketches_ratw.append(emb_ratw)
        sketches_st_starts.append(st_starts)
        # create stopbits
        stopbit = torch.zeros(emb_ctrlpt.shape[0], 1, device=device); stopbit[-1, 0] = 1.
        sketches_stopbits.append(stopbit)
        n_strokes.append(emb_ctrlpt.shape[0])
    
    n_strokes = torch.tensor(n_strokes, device=device)
    sketches_ctrlpt = pad_sequence(sketches_ctrlpt, batch_first=True)
    if embedder.rational:
        sketches_ratw = pad_sequence(sketches_ratw, batch_first=True)
    sketches_st_starts = pad_sequence(sketches_st_starts, batch_first=True)
    sketches_stopbits = pad_sequence(sketches_stopbits, batch_first=True, padding_value=1.0)

    # For every sketch in a batch:
    #   For every stroke in the sketch:
    #     1. (Control Point, Rational Weights) pair
    #     2. Start location of the stroke with respect to a global reference (of the sketch)
    if embedder.rational:
        return sketches_ctrlpt, sketches_ratw, sketches_st_starts, sketches_stopbits, n_strokes
    else:
        return sketches_ctrlpt, sketches_st_starts, sketches_stopbits, n_strokes

def inference(qdl, model, embedder, emblayers, embhidden, layers, hidden, n_mix, nsamples, rsamples, variational, bezier_degree, savefile, device):
    with torch.no_grad():
        fig, ax = plt.subplots(nsamples, (rsamples + 1), figsize=(rsamples * 8, nsamples * 4))
        for i, B in enumerate(qdl):

            h_initial_emb = torch.zeros(emblayers * 2, 256, embhidden, dtype=torch.float32)
            c_initial_emb = torch.zeros(emblayers * 2, 256, embhidden, dtype=torch.float32)
            h_initial = torch.zeros(layers * 2, 1, hidden, dtype=torch.float32)
            c_initial = torch.zeros(layers * 2, 1, hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                h_initial, h_initial_emb, c_initial, c_initial_emb = h_initial.cuda(), h_initial_emb.cuda(), c_initial.cuda(), c_initial_emb.cuda()

            with torch.no_grad():
                if model.rational:
                    ctrlpts, ratws, starts, _, n_strokes = stroke_embed(B, (h_initial_emb, c_initial_emb), embedder)
                else:
                    ctrlpts, starts, _, n_strokes = stroke_embed(B, (h_initial_emb, c_initial_emb), embedder)
                    ratws = torch.ones(ctrlpts.shape[0], ctrlpts.shape[1], model.n_ratw, device=ctrlpts.device)

                _cpad = torch.zeros(ctrlpts.shape[0], 1, ctrlpts.shape[2], device=device)
                _rpad = torch.zeros(ratws.shape[0], 1, ratws.shape[2], device=device)
                _spad = torch.zeros(starts.shape[0], 1, starts.shape[2], device=device)
                ctrlpts = torch.cat([_cpad, ctrlpts], dim=1)
                ratws = torch.cat([_rpad, ratws], dim=1)
                starts = torch.cat([_spad, starts], dim=1)

            for i in range(256):
                if i == nsamples:
                    break
                
                n_stroke = n_strokes[i]
                drawsketch(ctrlpts[i,1:n_stroke+1,:], ratws[i,1:n_stroke+1,:], starts[i,1:n_stroke+1,:], n_stroke, ax[i, 0])

                for r in range(rsamples):
                    if model.rational:
                        out_param_mu, out_param_std, out_param_mix, out_stopbits = model((h_initial, c_initial), 
                            ctrlpts[i,:n_stroke,:].unsqueeze(0), ratws[i,:n_stroke,:].unsqueeze(0), starts[i,:n_stroke,:].unsqueeze(0))
                    else:
                        out_param_mu, out_param_std, out_param_mix, out_stopbits = model((h_initial, c_initial), 
                            ctrlpts[i,:n_stroke,:].unsqueeze(0), None, starts[i,:n_stroke,:].unsqueeze(0))

                    # reshape to make the n_mix visible
                    out_param_mu = out_param_mu.view(1, -1, n_mix, out_param_mu.shape[-1]//n_mix)
                    out_param_std = out_param_std.view(1, -1, n_mix, out_param_std.shape[-1]//n_mix)

                    # sampled params at all timesteps
                    out_ctrlpts = torch.zeros(n_stroke, model.n_ctrlpt)
                    if model.rational:
                        out_ratws = torch.zeros(n_stroke, model.n_ratw)
                    else:
                        out_ratws = torch.ones(n_stroke, model.n_ratw)
                    out_starts = torch.zeros(n_stroke, model.n_start)

                    mix_ids = dist.Categorical(out_param_mix.squeeze()).sample()
                    for t, (mix_id, mu, std) in enumerate(zip(mix_ids, out_param_mu[0,...], out_param_std[0,...])):
                        mu, std = mu[mix_id.item(), :], std[mix_id.item(), :]
                        sample = dist.Normal(mu, std).sample()
                        out_ctrlpts[t, :] = sample[:model.n_ctrlpt]
                        if model.rational:
                            out_ratws[t, :] = sample[model.n_ctrlpt:model.n_ctrlpt+model.n_ratw]
                            out_starts[t, :] = sample[model.n_ctrlpt+model.n_ratw:]
                        else:
                            out_starts[t, :] = sample[model.n_ctrlpt:]
                    drawsketch(out_ctrlpts, out_ratws, out_starts, n_stroke, ax[i, 1+r])

            break # just one batch enough

        plt.xticks([]); plt.yticks([])
        plt.savefig(savefile)
        plt.close()