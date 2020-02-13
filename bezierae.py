import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

from bezierloss import BezierLoss

class RNNBezierAE(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_latent, bezier_degree_low, bezier_degree_high,
        dtype=torch.float32, bidirectional=True, dropout=0.8, rational=True):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden, self.n_layer = n_input, n_hidden, n_layer
        self.n_latent = n_latent
        
        self.bezier_degree = list(range(bezier_degree_low, bezier_degree_high + 1))
        self.n_latent_ctrl = [(z + 1 - 1) * 2 for z in self.bezier_degree] # The second '-1' is for Delta_P encoding
        self.n_latent_ratw = [z + 1 - 2 for z in self.bezier_degree]
        
        self.bidirectional = 2 if bidirectional else 1
        self.dtype = dtype
        self.dropout = dropout
        self.rational = rational

        # The t-network
        self.tcell = self.tcell = nn.LSTM(self.n_input, self.n_hidden, self.n_layer,
            bidirectional=bidirectional, dropout=self.dropout)

        self.t_logits = nn.ModuleList([torch.nn.Linear(self.bidirectional * self.n_hidden, 1) for _ in self.bezier_degree])

        # ...
        n_hc = 2 * self.bidirectional * self.n_hidden
        self.hc_project = nn.Linear(n_hc, self.n_latent)

        self.ctrlpt_arms = nn.ModuleList([nn.Linear(self.n_latent, c) for c in self.n_latent_ctrl])
        if self.rational:
            self.ratw_arms = nn.ModuleList([nn.Linear(self.n_latent, self.n_latent_ratw) for r in self.n_latent_ratw])

        # Bezier mechanics
        self.bezierlosses = nn.ModuleList([BezierLoss(z, reg_weight_p=None, reg_weight_r=None) for z in self.bezier_degree])

    def constraint_t(self, ts, lens):
        ts = ts.squeeze(-1)
        csm = []
        for t, l in zip(ts, lens):
            csm.append( torch.cumsum(torch.softmax(t[:l.item()], 0), 0) )
        csm = pad_sequence(csm, batch_first=True, padding_value=0.)
        return csm

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)

    def forward(self, x, h_initial, c_initial):
        out, (h_final, c_final) = self.tcell(x, (h_initial, c_initial))
        hns, lens = pad_packed_sequence(out, batch_first=True)
        
        t_logits = [t_logit(hns) for t_logit in self.t_logits]
        ts = [self.constraint_t(t_logit, lens) for t_logit in t_logits]


        # latent space
        h_final = h_final.view(self.n_layer, self.bidirectional, -1, self.n_hidden)
        c_final = c_final.view(self.n_layer, self.bidirectional, -1, self.n_hidden)
        H = torch.cat([h_final[-1, 0], h_final[-1, 1]], 1)
        C = torch.cat([c_final[-1, 0], c_final[-1, 0]], 1)
        HC = torch.cat([H, C], 1) # concat all "states" of the LSTM

        hc_projection = F.relu(self.hc_project(HC))
        latent_ctrlpt = [ctrlpt_arm(hc_projection) for ctrlpt_arm in self.ctrlpt_arms]

        # 'P's should be encoded as [P0=0, DelP1, DelP2, ..]
        # latent_ctrlpt = latent_ctrlpt.view(-1, self.n_latent_ctrl // 2, 2)
        latent_ctrlpt = [ctrlpt.view(-1, ctrlpt.shape[1] // 2, 2) for ctrlpt in latent_ctrlpt]
        latent_ctrlpt_return = latent_ctrlpt
        P0 = torch.zeros(latent_ctrlpt[0].shape[0], 1, 2, device=latent_ctrlpt[0].device)
        latent_ctrlpt = [torch.cat([P0, ctrlpt], 1) for ctrlpt in latent_ctrlpt]
        latent_ctrlpt = [torch.cumsum(ctrlpt, 1) for ctrlpt in latent_ctrlpt]
        # breakpoint()

        if self.rational:
            latent_ratw = [ratw_arm(hc_projection) for ratw_arm in self.ratw_arms]
            z_ = torch.ones((latent_ratw[0].shape[0], 1), device=latent_ratw[0].device) * 5. # sigmoid(5.) is close to 1
            latent_ratw_padded = [torch.cat([z_, ratw, z_], 1) for ratw in latent_ratw]
        
        if self.training:
            out, regu = [], []
            if self.rational:
                latent_ratw_padded_sigm = [torch.sigmoid(r) for r in latent_ratw_padded]
                for loss, z_ts, z_latent_ctrlpt, z_latent_ratw in zip(self.bezierlosses, ts, latent_ctrlpt, latent_ratw_padded_sigm):
                    z_out, z_regu = [], []
                    for t, p, r, l in zip(z_ts, z_latent_ctrlpt, z_latent_ratw, lens):
                        z_out.append( loss(p, r, None, ts=t[:l]) )
                        z_regu.append( (loss._consecutive_dist(p)**2).mean() )
                    out.append(z_out)
                    regu.append(z_regu)
            else:
                for loss, z_ts, z_latent_ctrlpt in zip(self.bezierlosses, ts, latent_ctrlpt):
                    z_out, z_regu = [], []
                    for t, p, l in zip(z_ts, z_latent_ctrlpt, lens):
                        z_out.append( loss(p, None, None, ts=t[:l]) )
                        z_regu.append( (loss._consecutive_dist(p)**2).mean() )
                    out.append(z_out)
                    regu.append(z_regu)
            
            return out, sum([sum(z_regu)/len(z_regu) for z_regu in regu]) / len(self.bezier_degree)

        else:
            if self.rational:
                return latent_ctrlpt_return, latent_ratw
            else:
                return latent_ctrlpt_return

class RNNSketchAE(nn.Module):
    def __init__(self, n_inps, n_hidden, n_layer = 2, n_mixture = 3, dropout = 0.8, eps = 1e-8, rational = True,
        variational = False, concatz = False):
        super().__init__()

        # Track parameters
        self.n_ctrlpt, self.n_ratw, self.n_start = n_inps
        self.n_hidden = n_hidden
        self.n_layer = 2
        self.n_hc = 2 * 2 * self.n_hidden
        self.n_latent = self.n_hc // 2
        self.dropout = dropout
        self.n_params = self.n_ctrlpt + (self.n_ratw if rational else 0) + self.n_start
        self.n_mixture = n_mixture
        self.rational = rational
        self.variational = variational
        self.concatz = concatz

        self.eps = eps

        # Layer definition
        self.encoder = nn.LSTM(self.n_params, self.n_hidden, self.n_layer, bidirectional=True, batch_first=True, dropout=dropout)
        if not self.concatz:
            self.decoder = nn.LSTM(self.n_params, 2 * self.n_hidden, self.n_layer, bidirectional=False, batch_first=True, dropout=dropout)
        else:
            self.decoder = nn.LSTM(self.n_params + self.n_latent, 2 * self.n_hidden, self.n_layer, bidirectional=False, batch_first=True, dropout=dropout)

        # Other transformations
        self.hc_to_latent = nn.Linear(self.n_hc, self.n_latent) # encoder side
        if self.variational:
            self.hc_to_latent_logvar = nn.Linear(self.n_hc, self.n_latent) # encoder side
        self.latent_to_h0_1 = nn.Linear(self.n_latent, self.n_hidden * 2) # decoder side
        self.latent_to_c0_1 = nn.Linear(self.n_latent, self.n_hidden * 2) # decoder side
        self.latent_to_h0_2 = nn.Linear(self.n_latent, self.n_hidden * 2) # decoder side
        self.latent_to_c0_2 = nn.Linear(self.n_latent, self.n_hidden * 2) # decoder side
        self.tanh = nn.Tanh()
        
        self.param_mu_arm = nn.Linear(self.n_hidden * 2, self.n_params * self.n_mixture)
        self.param_std_arm = nn.Linear(self.n_hidden * 2, self.n_params * self.n_mixture) # put through exp()
        self.param_mix_arm = nn.Linear(self.n_hidden * 2, self.n_mixture) # put through softmax
        self.stopbit_arm = nn.Linear(self.n_hidden * 2, 1)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)
    
    def forward(self, initials, ctrlpt, ratw, start, inference=False):
        h_initial, c_initial = initials
        if self.rational:
            input = torch.cat([ctrlpt, ratw, start], -1)
        else:
            input = torch.cat([ctrlpt, start], -1)
        _, (hn, cn) = self.encoder(input, (h_initial, c_initial))
        hn = hn.view(self.n_layer, 2, -1, self.n_hidden)
        cn = cn.view(self.n_layer, 2, -1, self.n_hidden)
        hn, cn = hn[-1,...], cn[-1,...] # only from the topmost layer

        hc = torch.cat([hn[0], hn[1], cn[0], cn[1]], -1) # concat all of 'em
        latent = self.hc_to_latent(hc)
        if self.variational:
            latent_mean = latent
            latent_logvar = self.hc_to_latent_logvar(hc)
            latent = self.reparam(latent, latent_logvar)

            KLD = -0.5 * torch.mean(1 + latent_logvar - latent.pow(2) - latent_logvar.exp())
        #### encoder ends here ####

        h01, c01 = self.latent_to_h0_1(latent), self.latent_to_c0_1(latent)
        h02, c02 = self.latent_to_h0_2(latent), self.latent_to_c0_2(latent)
        h0 = self.tanh(torch.stack([h01, h02], 0))
        c0 = self.tanh(torch.stack([c01, c02], 0))

        if self.concatz:
            latent_c = latent.view(-1, 1, self.n_latent).repeat(1, input.shape[1], 1)
            input = torch.cat([input, latent_c], -1)
        state, _ = self.decoder(input, (h0, c0))

        # out_ctrlpt = self.ctrlpt_arm(state)
        # out_ratw = self.ratw_arm(state)
        # out_start = self.start_arm(state)
        out_param_mu = self.param_mu_arm(state)
        out_param_std = torch.exp(self.param_std_arm(state))
        out_param_mix = torch.softmax(self.param_mix_arm(state), -1)
        out_stopbit = torch.sigmoid(self.stopbit_arm(state))

        if self.training:
            if not self.variational:
                return out_param_mu, out_param_std, out_param_mix, out_stopbit
            else:
                return out_param_mu, out_param_std, out_param_mix, out_stopbit, KLD
        else:
            
            if inference:
                L = input.shape[1] # just as a safety (see the for loop)
                input = torch.zeros(1, 1, self.n_params, device=input.device)
                stop = False

                out_ctrlpts, out_ratws, out_starts = [], [], []
                for _ in range(L):
                    if self.concatz:
                        latent_c = latent.view(1, 1, self.n_latent)
                        input = torch.cat([input, latent_c], -1)
                    state, (h1, c1) = self.decoder(input, (h0, c0))
                    
                    out_param_mu = self.param_mu_arm(state).squeeze()
                    out_param_std = torch.exp(self.param_std_arm(state)).squeeze()
                    out_param_mix = torch.softmax(self.param_mix_arm(state), -1).squeeze()
                    out_stopbit = torch.sigmoid(self.stopbit_arm(state)).squeeze()

                    # reshape to make the n_mix visible
                    out_param_mu = out_param_mu.view(self.n_mixture, out_param_mu.shape[-1] // self.n_mixture)
                    out_param_std = out_param_std.view(self.n_mixture, out_param_std.shape[-1] // self.n_mixture)

                    mix_id = dist.Categorical(out_param_mix.squeeze()).sample()

                    mu, std = out_param_mu[mix_id.item(), :], out_param_std[mix_id.item(), :]
                    sample = dist.Normal(mu, std).sample()
                    out_ctrlpts.append(sample[:self.n_ctrlpt])
                    if self.rational:
                        out_ratws.append(sample[self.n_ctrlpt:self.n_ctrlpt+self.n_ratw])
                        out_starts.append(sample[self.n_ctrlpt+self.n_ratw:])
                        input = torch.cat([out_ctrlpts[-1], out_ratws[-1], out_starts[-1]], -1)
                    else:
                        out_starts.append(sample[self.n_ctrlpt:])
                        input = torch.cat([out_ctrlpts[-1], out_starts[-1]], -1)

                    input = input.unsqueeze(0).unsqueeze(0)
                    h0, c0 = h1, c1

                    if out_stopbit.item() >= 0.99:
                        break
                
                out_ctrlpts = torch.stack(out_ctrlpts, 0)
                if self.rational:
                    out_ratws = torch.stack(out_ratws, 0)
                out_starts = torch.stack(out_starts, 0)

                if self.rational:
                    return out_ctrlpts, out_ratws, out_starts
                else:
                    return out_ctrlpts, out_starts
            
            if not self.variational:
                # as of now, teacher-frocing even in testing
                return out_param_mu, out_param_std, out_param_mix, out_stopbit
            else:
                return out_param_mu, out_param_std, out_param_mix, out_stopbit, KLD

# def gmm_loss(mu, std, mix, n_mix, ctrlpt, ratw, start):
#     param = torch.cat([ctrlpt, ratw, start], -1)
#     mus = torch.split(mu, mu.shape[-1]//n_mix, -1)
#     stds = torch.split(std, std.shape[-1]//n_mix, -1)
#     mixs = torch.split(mix, mix.shape[-1]//n_mix, -1)
#     Ns = [dist.Normal(m, s) for m, s in zip(mus, stds)]
#     pdfs = []
#     for N, pi in zip(Ns, mixs):
#         pdfs.append((N.log_prob(param).sum(-1).exp() + 1e-10) * pi.view(-1,))
#         breakpoint()
#     return -sum(pdfs).log().mean()

def gmm_loss(batch, mus, sigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
    # TAKEN FROM: https://github.com/ctallec/world-models/blob/master/models/mdrnn.py
    ## NOT MY CODE
    
    """ Computes the gmm loss.
    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.
    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited
    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))
    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    batch = batch.unsqueeze(-2)
    normal_dist = dist.Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob