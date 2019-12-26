import torch, pdb
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from bezierloss import BezierLoss

class RNNStrokeEncoder(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_latent, dtype=torch.float32, bidirectional=True, dropout=0.5, variational=True):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden = n_input, n_hidden
        self.n_layer = n_layer
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.dropout = dropout
        self.variational = variational
        self.n_latent = n_latent

        self.cell = nn.LSTM(self.n_input, self.n_hidden, self.n_layer, bidirectional=bidirectional, dropout=self.dropout)

        # The latent vector producer
        self.latent = nn.Linear(2 * self.n_hidden * 2, self.n_latent)

        if self.variational:
            self.logvar = nn.Linear(2 * self.n_hidden * 2, self.n_latent)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)

    def forward(self, x, h_initial, c_initial, KLD_anneal = 1):
        _, (h_final, c_final) = self.cell(x, (h_initial, c_initial))
        H = torch.cat([h_final[0], h_final[1]], dim=1)
        C = torch.cat([c_final[0], h_final[1]], dim=1)
        HC = torch.cat([H, C], dim=1)

        if not self.variational:
            return self.latent(HC)
        else:
            mu = self.latent(HC)
            logvar = self.logvar(HC) * KLD_anneal

            # KL divergence term of the loss
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            if self.training:
                return self.reparam(mu, logvar), KLD
            else:
                return mu, torch.exp(0.5 * logvar)

class RNNStrokeDecoder(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_latent, n_output, dtype=torch.float32, bidirectional=True, dropout=0.5):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden, self.n_output = n_input, n_hidden, n_output
        self.n_layer = n_layer
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.dropout = dropout
        self.n_latent = n_latent

        self.cell = nn.LSTM(self.n_input, self.n_hidden, self.n_layer, bidirectional=bidirectional, dropout=dropout)

        # project the latent into (H0, C0) using these
        self.H = nn.Linear(self.n_latent, self.n_hidden)
        self.C = nn.Linear(self.n_latent, self.n_hidden)

        # Output layer
        self.ctrlpt_mu = torch.nn.Linear(self.n_hidden * self.bidirectional, self.n_output)
        self.ctrlpt_std = torch.nn.Linear(self.n_hidden * self.bidirectional, self.n_output)
        # Using Rational Bezier curve
        self.ratw_mu = torch.nn.Linear(self.n_hidden * self.bidirectional, 1) # rational weights mean
        # self.ratw_std = torch.nn.Linear(self.n_hidden * self.bidirectional, 1) # rational weights sigma

    def forward(self, x, latent):
        # H_f, H_b = torch.split(h_initial, [self.n_hidden, self.n_hidden], dim=1)
        # h_initial = torch.stack([H_f, H_b], dim=0)
        h_initial = self.H(latent).unsqueeze(0)
        c_initial = self.C(latent).unsqueeze(0)
        # breakpoint()

        out, (_, _) = self.cell(x, (h_initial, c_initial))
        hns, lengths = pad_packed_sequence(out, batch_first=True)
        # hns = hns.permute(1, 0, 2) # Make it batch first
        # breakpoint()
        
        out_ctrlpt, out_ratw = [], []
        for hn, l in zip(hns, lengths):
            h = hn[:l, :]
            ctrlpt_mu = self.ctrlpt_mu(h)
            ctrlpt_std = self.ctrlpt_std(h)
            ratw_mu = self.ratw_mu(h[1:-1]).squeeze()
            # ratw_std = self.ratw_std(h[1:-1]).squeeze()
            ratw_mu = torch.tensor([0., *ratw_mu, 0.], device=ratw_mu.device)
            # ratw_std = torch.tensor([0., *ratw_std, 0.], device=ratw_std.device)
            ratw_mu = torch.sigmoid(ratw_mu)
            # breakpoint()

            c = torch.distributions.Normal(ctrlpt_mu, ctrlpt_std)
            # r = torch.distributions.Normal(ratw_mu, ratw_std)
            out_ctrlpt.append(c.rsample())
            out_ratw.append(ratw_mu)
        
        return out_ctrlpt, out_ratw

class RNNStrokeAE(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_output, n_latent, dtype=torch.float32, bidirectional=True,
                bezier_degree=0, dropout=0.5, variational=False):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden, self.n_output = n_input, n_hidden, n_output
        self.n_layer = n_layer
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.dropout = dropout
        self.bezier_degree = bezier_degree
        self.variational = variational
        self.n_latent = n_latent

        self.encoder = RNNStrokeEncoder(self.n_input, self.n_hidden, self.n_layer, self.n_latent, variational=self.variational,
            dtype=self.dtype, bidirectional=bidirectional, dropout=self.dropout)
        # Decoder is always twice deep the encoder because of bi-uni nature of enc-dec
        self.decoder = RNNStrokeDecoder(self.n_input, self.n_hidden, self.n_layer, self.n_latent, self.n_output,
            dtype=self.dtype, bidirectional=False, dropout=self.dropout)


    def forward(self, x, h_initial, c_initial, KLD_anneal = 1.):
        if not self.variational:
            latent = self.encoder(x, h_initial, c_initial)
        else:
            if self.training:
                latent, KLD = self.encoder(x, h_initial, c_initial, KLD_anneal=KLD_anneal)
            else:
                latent, std = self.encoder(x, h_initial, c_initial)

        # x, l = pad_packed_sequence(x, batch_first=True)
        # x = torch.zeros_like(x) # Input free decoder
        # if self.bezier_degree != 0:
        #     # Bezier curve should have output sequence
        #     # length always equal to 'self.bezier_degree'
        #     x = x[:,:self.bezier_degree + 1,:]
        #     l = torch.ones_like(l) * (self.bezier_degree + 1)
        # else:
        #     raise 'Bezier curve with degree zero not possible'
        batch_size = x.batch_sizes.max()
        x = torch.zeros((batch_size, self.bezier_degree + 1, self.n_input), device=x.data.device)
        l = torch.ones((batch_size,), device=x.data.device) * (self.bezier_degree + 1)
        
        x = pack_padded_sequence(x, l, enforce_sorted=False, batch_first=True)
        # breakpoint()

        out_ctrlpt, out_ratw = self.decoder(x, latent)

        if self.variational:
            if self.training:
                return latent, (out_ctrlpt, out_ratw), KLD
            else:
                return (out_ctrlpt, out_ratw), std
        else:
            if self.training:
                return latent, (out_ctrlpt, out_ratw)
            else:
                return (out_ctrlpt, out_ratw)

class StrokeMSELoss(nn.Module):
    def __init__(self, bezier_degree, n_latent, min_stroke_len=2, bez_reg_weight_p=1e-2, bez_reg_weight_r=1e-2):
        super().__init__()

        # Track the parameters
        self.min_stroke_len = min_stroke_len
        self.bezier_degree = bezier_degree
        self.n_latent = n_latent

        # standard MSELoss
        self.mseloss = nn.MSELoss()
        self.bezierloss = BezierLoss(self.bezier_degree, reg_weight_p=bez_reg_weight_p, reg_weight_r=bez_reg_weight_r)

        # global t-estimator
        self.tcell = nn.GRU(2, self.n_latent, 1, bidirectional=False)
        self.tarm = nn.Linear(self.n_latent, 1)

    def forward(self, out_ctrlpt, out_ratw, xy, lens, latent):
        loss = []

        packed_xy = pack_padded_sequence(xy, lens, batch_first=True, enforce_sorted=False)
        t, _ = self.tcell(packed_xy, latent.unsqueeze(0))
        t, _ = pad_packed_sequence(t, batch_first=True)
        t = self.tarm(t).squeeze()

        for q, (y_ctrlpt, y_ratw, y, l, t_logit, lat) in enumerate(zip(out_ctrlpt, out_ratw, xy, lens, t, latent)):
            if l >= 2 and (y.sum().item() != 0.0):
                y, t_logit = y[:l.item(),:], t_logit[:l.item()]
                # bezierloss_x = getattr(self, f'bezierloss_{q}')
                t_actual = torch.cumsum(torch.softmax(t_logit, 0), 0)
                loss.append( self.bezierloss(y_ctrlpt, y_ratw, y, ts=t_actual) )

        return sum(loss) / len(loss)


class BezierFunction(nn.Module):
    def __init__(self, n_input, n_output = 2):
        super().__init__()

        # Track parameters
        self.n_input = n_input
        self.n_output = n_output
        self.n_interm = (self.n_input + self.n_output) // 2

        # The model
        self.model = nn.Sequential(
            nn.Linear(self.n_input, self.n_interm),
            nn.ReLU(),
            nn.Linear(self.n_interm, self.n_output)
        )

    def forward(self, x):
        # TODO: forward pass of \overline{C}([t, \Theta])
        return self.model(x)

class RNNBezierDecoder(nn.Module):
    def __init__(self, n_input, n_latent, n_layer, dtype=torch.float32, dropout=0.5):
        super().__init__()

        # Track parameters
        self.n_input, self.n_latent = n_input, n_latent
        self.n_layer = n_layer
        self.dtype = dtype
        self.dropout = dropout
        self.n_t_enc = self.n_latent

        self.cell = nn.GRU(self.n_input, self.n_latent, self.n_layer, bidirectional=False, dropout=dropout)

        # Output layer
        self.t_logits = torch.nn.Linear(self.n_latent, 1)
        self.t_enc = torch.nn.Linear(1, self.n_t_enc)
        self.C = BezierFunction(self.n_latent + self.n_t_enc)

    def constraint_t(self, t):
        return torch.cumsum(torch.softmax(t.squeeze(), 1), 1).unsqueeze(-1)

    def forward(self, x, h_initial):
        if self.training:
            out, _ = self.cell(x, h_initial.unsqueeze(0))
            out, l = pad_packed_sequence(out, batch_first=True)
            t = self.t_logits(out)
            t = self.constraint_t(t)
            breakpoint()
        else:
            t = x
        t_enc = self.t_enc(t)
        h_context = h_initial.unsqueeze(1).repeat(1, t.shape[1], 1)
        t_enc_h_context = torch.cat([t_enc, h_context], -1)
        out = self.C(t_enc_h_context)
        return out

class RNNBezierAE(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_latent, dtype=torch.float32, bidirectional=True, dropout=0.5, variational=False):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden = n_input, n_hidden
        self.n_layer = n_layer
        self.n_latent = n_latent
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.dropout = dropout
        self.variational = variational

        self.encoder = RNNStrokeEncoder(self.n_input, self.n_hidden, self.n_layer, self.n_latent, variational=self.variational,
            dtype=self.dtype, bidirectional=bidirectional, dropout=self.dropout)
        
        self.decoder = RNNBezierDecoder(self.n_input, self.n_latent, 1, dtype=self.dtype, dropout=self.dropout)

    def forward(self, x, h_initial):
        if not self.variational:
            latent = self.encoder(x, h_initial)
        else:
            if self.training:
                latent, KLD = self.encoder(x, h_initial)
            else:
                latent, std = self.encoder(x, h_initial)

        out = self.decoder(x, latent)

        if self.variational:
            if self.training:
                return out, KLD
            else:
                return out, std
        else:
            return out

if __name__ == '__main__':
    model = RNNStrokeAE(3, 256, 3, 2)
    print(model)