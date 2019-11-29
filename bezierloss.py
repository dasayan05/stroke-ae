import torch, numpy as np
import torch.nn as nn
from beziermatrix import bezier_matrix

class BezierLoss(nn.Module):
    def __init__(self, degree, n_xy = None, reg_weight = 1e-2):
        super().__init__()
        self.degree = degree
        self.M = self._M(self.degree)
        if torch.cuda.is_available():
            self.M = self.M.cuda()
        self.reg_weight = reg_weight
        self.n_xy = n_xy

        if self.n_xy != None:
            # The t-estimators
            self.tarm = nn.Linear(2 * self.n_xy, self.n_xy)
            if torch.cuda.is_available():
                self.tarm = self.tarm.cuda()

    def _consecutive_dist(self, XY):
        return (((XY[1:,:] - XY[0:-1,:])**2).sum(axis=1))**0.5

    def _heuristic_ts(self, XY):
        ds = self._consecutive_dist(XY)
        ds = ds / ds.sum()
        # breakpoint()
        return torch.cumsum(torch.tensor([0., *ds]), 0)

    def _T(self, ts, d, dtype=torch.float32):
        # breakpoint()
        ts = ts[..., np.newaxis]
        Q = [ts**n for n in range(d, -1, -1)]
        Q = torch.cat(Q, 1)
        if torch.cuda.is_available():
            Q = Q.cuda()
        return Q

    def _M(self, d: 'degree'):
        return torch.tensor(bezier_matrix(d), dtype=torch.float32)

    def forward(self, P, XY):
        if self.n_xy != None:
            assert self.n_xy == XY.shape[0]
        
        if self.n_xy != None:
            logits = self.tarm(XY.view(-1, self.n_xy * 2)).squeeze()
            sm = torch.softmax(logits, 0)
            ts = torch.cumsum(sm, 0)
        else:
            ts = self._heuristic_ts(XY)

        C = torch.mm(self._T(ts, self.degree), torch.mm(self.M, P))
        l = ((C - XY)**2).mean() + self.reg_weight * (self._consecutive_dist(P)**2).mean()
        return l