import torch, numpy as np
import torch.nn as nn
from beziermatrix import bezier_matrix

class BezierLoss(nn.Module):
    def __init__(self, degree, reg_weight_p = 1e-2, reg_weight_r = 1e-2):
        super().__init__()
        self.degree = degree
        self.M = self._M(self.degree)
        if torch.cuda.is_available():
            self.M = self.M.cuda()
        self.reg_weight_p = reg_weight_p
        self.reg_weight_r = reg_weight_r

    def _consecutive_dist(self, XY):
        return (((XY[1:,:] - XY[0:-1,:])**2).sum(axis=1))**0.5

    def _heuristic_ts(self, XY):
        ds = self._consecutive_dist(XY)
        ds = ds / ds.sum()
        return torch.cumsum(torch.tensor([0., *ds]), 0)

    def _T(self, ts, d, dtype=torch.float32):
        ts = ts[..., np.newaxis]
        Q = [ts**n for n in range(d, -1, -1)]
        Q = torch.cat(Q, 1)
        if torch.cuda.is_available():
            Q = Q.cuda()
        return Q

    def _M(self, d: 'degree'):
        return torch.tensor(bezier_matrix(d), dtype=torch.float32)

    def forward(self, P, R, XY, ts=None):
        # breakpoint()
        if R is not None:
            C = torch.mm(self._T(ts, self.degree), torch.mm(self.M, torch.diag(R)))
            C = C / C.sum(1).unsqueeze(1)
            C = torch.mm(C, P)
        else:
            C = torch.mm(self._T(ts, self.degree), torch.mm(self.M, P))

        if XY is None:
            return C
        else:
            l = ((C - XY)**2).mean() + self.reg_weight_p * (self._consecutive_dist(P)**2).mean()
            return l