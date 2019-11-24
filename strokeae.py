import torch, pdb
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class RNNStrokeEncoder(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, dtype=torch.float32, bidirectional=True, dropout=0.5):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden = n_input, n_hidden
        self.n_layer = n_layer
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.dropout = dropout

        self.cell = nn.GRU(self.n_input, self.n_hidden, self.n_layer, bidirectional=bidirectional, dropout=self.dropout)

    def forward(self, x, h_initial):
        _, h_final = self.cell(x, h_initial)
        return h_final

class RNNStrokeDecoder(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_output, dtype=torch.float32, bidirectional=True, dropout=0.5):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden, self.n_output = n_input, n_hidden, n_output
        self.n_layer = n_layer
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.dropout = dropout

        self.cell = nn.GRU(self.n_input, self.n_hidden, self.n_layer, bidirectional=bidirectional, dropout=dropout)

        # Output layer
        self.next = torch.nn.Linear(self.n_hidden * self.bidirectional, self.n_output)
        self.p = torch.nn.Linear(self.n_hidden * self.bidirectional, 1)

    def forward(self, x, h_initial):
        out, _ = self.cell(x, h_initial)
        hns, lengths = pad_packed_sequence(out, batch_first=True)
        # pdb.set_trace()
        
        out, P = [], []
        for hn, l in zip(hns, lengths):
            h = hn[:l, :]
            out.append(self.next(h))
            P.append(torch.sigmoid(self.p(h)))
        
        return out, P

class RNNStrokeAE(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_output, dtype=torch.float32, bidirectional=True,
                ip_free_decoding=False, dropout=0.5):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden, self.n_output = n_input, n_hidden, n_output
        self.n_layer = n_layer
        self.dtype = dtype
        self.bidirectional = 2 if bidirectional else 1
        self.dropout = dropout
        self.ip_free_decoding = ip_free_decoding

        self.encoder = RNNStrokeEncoder(self.n_input, self.n_hidden, self.n_layer,
            dtype=self.dtype, bidirectional=bidirectional, dropout=self.dropout)
        self.decoder = RNNStrokeDecoder(self.n_input, self.n_hidden, self.n_layer, self.n_output,
            dtype=self.dtype, bidirectional=bidirectional, dropout=self.dropout)

    def forward(self, x, x_, h_initial):
        latent = self.encoder(x, h_initial)
        if self.ip_free_decoding:
            x_, l = pad_packed_sequence(x_)
            x_ = torch.zeros_like(x_) # Input free decoder
            x_ = pack_padded_sequence(x_, l, enforce_sorted=False)
        out, P = self.decoder(x_, latent)
        return out, P

class StrokeMSELoss(nn.Module):
    def __init__(self, min_stroke_len=2):
        super().__init__()

        # Track the parameters
        self.min_stroke_len = min_stroke_len

        # standard MSELoss
        self.mseloss = nn.MSELoss()

    def forward(self, out_xy, out_p, xy, p, lens):
        loss = []

        for y_, p_, y, p, l in zip(out_xy, out_p, xy, p, lens):
            if l >= 2:
                y, p = y[:l.item(),:], p[:l.item()]
                loss.append( self.mseloss(y, y_) )
                loss.append( self.mseloss(p, p_.squeeze()) )

        return sum(loss) / len(loss)


if __name__ == '__main__':
    model = RNNStrokeAE(3, 256, 3, 2)
    print(model)