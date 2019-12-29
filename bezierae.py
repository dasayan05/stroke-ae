import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from bezierloss import BezierLoss

class RNNBezierAE(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, bezier_degree, dtype=torch.float32, bidirectional=True,
        variational=False, dropout=0.8):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden, self.n_layer = n_input, n_hidden, n_layer
        self.bezier_degree = bezier_degree
        self.n_latent_ctrl = (self.bezier_degree + 1) * 2
        self.n_latent_ratw = self.bezier_degree + 1 - 2
        self.bidirectional = 2 if bidirectional else 1
        self.dtype = dtype
        self.variational = variational
        self.dropout = dropout

        # The t-network
        self.tcell = self.tcell = nn.LSTM(self.n_input, self.n_hidden, self.n_layer,
            bidirectional=bidirectional, dropout=self.dropout)
        self.t_logits = torch.nn.Linear(self.bidirectional * self.n_hidden, 1)

        # ...
        n_hc = 2 * self.bidirectional * self.n_hidden
        n_project = (n_hc + self.n_latent_ctrl) // 2
        self.hc_project = nn.Linear(n_hc, n_project)

        self.ctrlpt_arm = nn.Linear(n_project, self.n_latent_ctrl)
        if self.variational:
            self.ctrlpt_logvar_arm = nn.Linear(n_project, self.n_latent_ctrl)
        self.ratw_arm = nn.Linear(n_project, self.n_latent_ratw)

        # Bezier mechanics
        self.bezierloss = BezierLoss(self.bezier_degree, reg_weight_p=None, reg_weight_r=None)

    def constraint_t(self, t):
        return torch.cumsum(torch.softmax(t.squeeze(-1), 1), 1)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)

    def forward(self, x, h_initial, c_initial):
        out, (h_final, c_final) = self.tcell(x, (h_initial, c_initial))
        hns, lens = pad_packed_sequence(out, batch_first=True)
        t_logits = self.t_logits(hns)
        ts = self.constraint_t(t_logits)

        # latent space
        h_final = h_final.view(self.n_layer, self.bidirectional, -1, self.n_hidden)
        c_final = c_final.view(self.n_layer, self.bidirectional, -1, self.n_hidden)
        H = torch.cat([h_final[-1, 0], h_final[-1, 1]], 1)
        C = torch.cat([c_final[-1, 0], c_final[-1, 0]], 1)
        HC = torch.cat([H, C], 1) # concat all "states" of the LSTM

        hc_projection = F.relu(self.hc_project(HC))
        latent_ctrlpt = self.ctrlpt_arm(hc_projection)

        # variational
        if self.variational:
            latent_ctrlpt_mean = latent_ctrlpt
            latent_ctrlpt_logvar = self.ctrlpt_logvar_arm(hc_projection)
            latent_ctrlpt = self.reparam(latent_ctrlpt, latent_ctrlpt_logvar)

            if self.training:
                KLD = -0.5 * torch.mean(1 + latent_ctrlpt_logvar - latent_ctrlpt.pow(2) - latent_ctrlpt_logvar.exp())

        latent_ctrlpt = latent_ctrlpt.view(-1, self.n_latent_ctrl // 2, 2)
        latent_ratw = self.ratw_arm(hc_projection)
        z_ = torch.zeros((latent_ratw.shape[0], 1), device=latent_ratw.device)
        latent_ratw = torch.cat([z_, latent_ratw, z_], 1)
        latent_ratw = torch.sigmoid(latent_ratw)
        
        out, regu = [], []
        for t, p, r, l in zip(ts, latent_ctrlpt, latent_ratw, lens):
            out.append( self.bezierloss(p, r, None, ts=t[:l]) )
            regu.append( (self.bezierloss._consecutive_dist(p)**2).mean() )
        
        if self.training:
            if not self.variational:
                return out, sum(regu) / len(regu)
            else:
                return out, sum(regu) / len(regu), KLD
        else:
            if not self.variational:
                return latent_ctrlpt, latent_ratw
            else:
                return latent_ctrlpt_mean, torch.exp(0.5 * latent_ctrlpt_logvar), latent_ratw