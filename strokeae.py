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

        self.cell = nn.GRU(self.n_input, self.n_hidden, self.n_layer, bidirectional=bidirectional, dropout=self.dropout)

        # The latent vector producer
        self.latent = nn.Linear(2 * self.n_hidden, self.n_latent)

        if self.variational:
            self.logvar = nn.Linear(2 * self.n_hidden, self.n_latent)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)

    def forward(self, x, h_initial, KLD_anneal = 1):
        # breakpoint()
        _, h_final = self.cell(x, h_initial)
        H = torch.cat([h_final[0], h_final[1]], dim=1)
        if not self.variational:
            return self.latent(H)
        else:
            mu = self.latent(H)
            logvar = self.logvar(H) * KLD_anneal

            # KL divergence term of the loss
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            if self.training:
                return self.reparam(mu, logvar), KLD
            else:
                return mu

class RNNStrokeDecoder(nn.Module):
    def __init__(self, n_input, n_layer, n_output, n_latent, dtype=torch.float32, bidirectional=True, dropout=0.5):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden, self.n_output = n_input, n_latent // 2, n_output
        self.n_layer = n_layer
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.dropout = dropout
        self.n_latent = n_latent

        self.cell = nn.GRU(self.n_input, self.n_hidden, self.n_layer, bidirectional=bidirectional, dropout=dropout)

        # Output layer
        self.next = torch.nn.Linear(self.n_hidden * self.bidirectional, self.n_output)
        self.p = torch.nn.Linear(self.n_hidden * self.bidirectional, 1)

    def forward(self, x, h_initial, return_state=False):
        H_f, H_b = torch.split(h_initial, [self.n_hidden, self.n_hidden], dim=1)
        h_initial = torch.stack([H_f, H_b], dim=0)

        out, state = self.cell(x, h_initial)
        hns, lengths = pad_packed_sequence(out, batch_first=True)
        
        out, P = [], []
        for hn, l in zip(hns, lengths):
            h = hn[:l, :]
            out.append(self.next(h))
            P.append(torch.sigmoid(self.p(h)))
        
        if not return_state:
            return out, P
        else:
            state = torch.cat([state[0], state[1]], dim=1)
            return (out, P), state

class RNNStrokeAE(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_output, n_latent, dtype=torch.float32, bidirectional=True,
                ip_free_decoding=False, bezier_degree=0, dropout=0.5, variational=False):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden, self.n_output = n_input, n_hidden, n_output
        self.n_layer = n_layer
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.dropout = dropout
        self.ip_free_decoding = ip_free_decoding
        self.bezier_degree = bezier_degree
        self.variational = variational
        self.n_latent = n_latent

        self.encoder = RNNStrokeEncoder(self.n_input, self.n_hidden, self.n_layer, self.n_latent, variational=self.variational,
            dtype=self.dtype, bidirectional=bidirectional, dropout=self.dropout)
        # Decoder is always twice deep the encoder because of bi-uni nature of enc-dec
        self.decoder = RNNStrokeDecoder(self.n_input, self.n_layer * 2, self.n_output, self.n_latent,
            dtype=self.dtype, bidirectional=False, dropout=self.dropout)


    def forward(self, x, x_, h_initial, KLD_anneal = 1.):
        if not self.variational:
            latent = self.encoder(x, h_initial)
        else:
            if self.training:
                latent, KLD = self.encoder(x, h_initial, KLD_anneal=KLD_anneal)
            else:
                latent = self.encoder(x, h_initial)

        if self.ip_free_decoding:
            x_, l = pad_packed_sequence(x_)
            x_ = torch.zeros_like(x_) # Input free decoder
            if self.bezier_degree != 0:
                # Bezier curve should have output sequence
                # length always equal to 'self.bezier_degree'
                x_ = x_[:self.bezier_degree + 1,:,:]
                l = torch.ones_like(l) * (self.bezier_degree + 1)
            x_ = pack_padded_sequence(x_, l, enforce_sorted=False)
        out, P = self.decoder(x_, latent)

        if self.variational:
            if self.training:
                return (out, P), KLD
            else:
                return out, P
        else:
            return out, P

class StrokeMSELoss(nn.Module):
    def __init__(self, XY_lens, min_stroke_len=2, bezier_degree=0, bez_reg_weight=1e-2):
        super().__init__()

        # Track the parameters
        self.min_stroke_len = min_stroke_len
        self.bezier_degree = bezier_degree
        self.XY_lens = XY_lens

        # standard MSELoss
        self.mseloss = nn.MSELoss()
        if self.bezier_degree != 0:
            for q, xylen in enumerate(self.XY_lens):
                setattr(self, f'bezierloss_{q}', BezierLoss(self.bezier_degree, n_xy=xylen, reg_weight=bez_reg_weight))
            # The fixed 'pen-up' prob ground-truth for bezier case
            # i.e., [0., 0., 0., .. , 1.] with no. of elements
            # equal to (self.bezier_degree + 1)
            self.bezier_p = torch.zeros((self.bezier_degree + 1))
            self.bezier_p[-1] = 1.
            if torch.cuda.is_available():
                self.bezier_p = self.bezier_p.cuda()

    def forward(self, out_xy, out_p, xy, p, lens):
        loss = []

        for q, (y_, p_, y, p, l) in enumerate(zip(out_xy, out_p, xy, p, lens)):
            if l >= 2:
                y, p = y[:l.item(),:], p[:l.item()]
                if self.bezier_degree != 0:
                    bezierloss_x = getattr(self, f'bezierloss_{q}')
                    loss.append( bezierloss_x(y_, y) )
                    # loss.append( self.mseloss(p_, self.bezier_p) )
                else:
                    loss.append( self.mseloss(y, y_) )
                    loss.append( self.mseloss(p, p_.squeeze()) )

        return sum(loss) / len(loss)


if __name__ == '__main__':
    model = RNNStrokeAE(3, 256, 3, 2)
    print(model)