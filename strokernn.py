import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

class StrokeRNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_layer, n_output, dropout=0.5):
        super().__init__()

        # Track parameters
        self.n_input, self.n_hidden, self.n_output = n_input, n_hidden, n_output
        self.n_layer = n_layer

        self.cell = nn.GRU(self.n_input, self.n_hidden, self.n_layer, dropout=dropout)

        # Output layer
        self.next = nn.Linear(self.n_hidden, self.n_output)
        self.p = nn.Linear(self.n_hidden, 1)

    def forward(self, x, h_initial):
        out, state = self.cell(x, h_initial)
        hns, lengths = pad_packed_sequence(out, batch_first=True)
        
        out, P = [], []
        for hn, l in zip(hns, lengths):
            h = hn[:l, :]
            out.append(self.next(h))
            P.append(torch.sigmoid(self.p(h)))
        
        if self.training:
            return (out, P), lengths
        else:
            return (out, P), state