import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from bezierloss import BezierLoss

class RNNBezierEncoder(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_latent, dtype=torch.float32, bidirectional=True, variational=True):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden = n_input, n_hidden
        self.n_layer = n_layer
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.variational = variational
        self.n_latent = n_latent

        self.cell = nn.LSTM(self.n_input, self.n_hidden, self.n_layer, bidirectional=bidirectional)

        # The latent vector producer
        self.latent_ctrlpt = nn.Linear(2 * self.n_hidden * 2, self.n_latent)
        self.latent_ratw = nn.Linear(2 * self.n_hidden * 2, self.n_latent // 2)

        if self.variational:
            self.latent_ctrlpt_logvar = nn.Linear(2 * self.n_hidden * 2, self.n_latent)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)

    def forward(self, x, h_initial, c_initial):
        _, (h_final, c_final) = self.cell(x, (h_initial, c_initial))
        H = torch.cat([h_final[0], h_final[1]], dim=1)
        C = torch.cat([c_final[0], h_final[1]], dim=1)
        HC = torch.cat([H, C], dim=1)

        # predict (control point, rational weight) pair
        latent_ctrlpt = self.latent_ctrlpt(HC)
        latent_ratw = torch.sigmoid(self.latent_ratw(HC))

        if not self.variational:
            return latent_ctrlpt, latent_ratw
        else:
            latent_ctrlpt_mu = latent_ctrlpt
            latent_ctrlpt_logvar = self.latent_ctrlpt_logvar(HC)

            # KL divergence term of the loss
            KLD = -0.5 * torch.mean(1 + latent_ctrlpt_logvar - latent_ctrlpt_mu.pow(2) - latent_ctrlpt_logvar.exp())

            if self.training:
                return self.reparam(latent_ctrlpt_mu, latent_ctrlpt_logvar), latent_ratw, KLD
            else:
                return latent_ctrlpt_mu, torch.exp(0.5 * latent_ctrlpt_logvar), latent_ratw

class RNNBezierDecoder(nn.Module):
    def __init__(self, n_input, n_hidden, n_latent, n_layer, dtype=torch.float32):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden, self.n_latent = n_input, n_hidden, n_latent
        self.n_layer = n_layer
        self.dtype = dtype

        self.t_hf = nn.Linear(self.n_latent + self.n_latent // 2, self.n_hidden)
        self.t_hb = nn.Linear(self.n_latent + self.n_latent // 2, self.n_hidden)
        self.t_cf = nn.Linear(self.n_latent + self.n_latent // 2, self.n_hidden)
        self.t_cb = nn.Linear(self.n_latent + self.n_latent // 2, self.n_hidden)
        self.tanh = nn.Tanh()

        self.tcell = nn.LSTM(self.n_input, self.n_hidden, self.n_layer, bidirectional=True)
        self.t_logits = torch.nn.Linear(2 * self.n_hidden, 1)

        self.bezierloss = BezierLoss((self.n_latent // 2) - 1, reg_weight_p=None, reg_weight_r=None)

    def constraint_t(self, t):
        return torch.cumsum(torch.softmax(t.squeeze(), 1), 1).unsqueeze(-1)

    def forward(self, x, latent_ctrlpt, latent_ratw):
        latent = torch.cat([latent_ctrlpt, latent_ratw], 1)
        # project the latent into (h0, c0) for forward and backward
        h0 = self.tanh(torch.stack([self.t_hf(latent), self.t_hb(latent)], 0))
        c0 = self.tanh(torch.stack([self.t_cf(latent), self.t_cb(latent)], 0))

        out, _ = self.tcell(x, (h0, c0))
        out, l = pad_packed_sequence(out, batch_first=True)
        ts = self.t_logits(out)
        ts = self.constraint_t(ts)

        latent_ctrlpt_as_p = latent_ctrlpt.view(-1, self.n_latent // 2, 2)
        latent_ratw_as_r = latent_ratw.view(-1, self.n_latent // 2)

        out = []
        reg_consec_dist = []
        for t, p, r in zip(ts.squeeze(), latent_ctrlpt_as_p, latent_ratw_as_r):
            # breakpoint()
            out.append( self.bezierloss(p, r, None, ts=t) )
            reg_consec_dist.append( (self.bezierloss._consecutive_dist(p)**2).mean() )

        return out, sum(reg_consec_dist)/len(reg_consec_dist)

class RNNBezierAE(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, bezier_degree, dtype=torch.float32, bidirectional=True, variational=False):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden = n_input, n_hidden
        self.n_layer = n_layer
        self.bezier_degree = bezier_degree
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.variational = variational

        self.encoder = RNNBezierEncoder(self.n_input, self.n_hidden, self.n_layer, 2 * (self.bezier_degree + 1), variational=self.variational, dtype=self.dtype, bidirectional=bidirectional)
        
        self.decoder = RNNBezierDecoder(self.n_input, self.n_hidden, 2 * (self.bezier_degree + 1), 1, dtype=self.dtype)

    def forward(self, x, h_initial, c_initial):
        if not self.variational:
            latent_ctrlpt, latent_ratw = self.encoder(x, h_initial, c_initial)
        else:
            if self.training:
                latent_ctrlpt, latent_ratw, KLD = self.encoder(x, h_initial, c_initial)
            else:
                latent_ctrlpt_mu, latent_ctrlpt_std, latent_ratw = self.encoder(x, h_initial, c_initial)

        if self.training:
            out, regu = self.decoder(x, latent_ctrlpt, latent_ratw)
            if self.variational:
                return out, regu, KLD
            else:
                return out, regu

        else:
            if self.variational:
                return latent_ctrlpt_mu, latent_ctrlpt_std, latent_ratw
            else:
                return latent_ctrlpt, latent_ratw