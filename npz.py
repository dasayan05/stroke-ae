import torch
import numpy as np

class NPZWriter(object):
    def __init__(self, filepath):
        super().__init__()
        # Track parameters
        self.filepath = filepath
        # internal list
        self.tr, self.ts, self.vl = [], [], []

    def add(self, ctrlpt_batch, start_batch, n_strokes):
        for ctrlpt, start, n_stroke in zip(ctrlpt_batch, start_batch, n_strokes):
            ctrlpts = torch.unbind(ctrlpt[:n_stroke.item()], dim=0)
            starts = torch.unbind(start[:n_stroke.item()], dim=0)
            
            # populate this
            sketch = np.empty((0, 3), dtype=np.float32)

            for c, s in zip(ctrlpts, starts):
                c = c.detach().cpu().numpy().reshape((-1, 2))
                s = s.detach().cpu().numpy()

                P0 = np.array([[0., 0.]]) # start P
                c = np.cumsum(np.concatenate((P0, c), 0), 0)
                c = c + s

                q = np.zeros((c.shape[0], 1), dtype=np.float32); q[-1, 0] = 1.

                sketch = np.vstack((sketch, np.hstack((c, q))))

            sketch[:,:2] *= 255.
            sketch = sketch.astype(np.int16)
            sketch[:,:2] -= sketch[0,:2]
            sketch[1:,:2] -= sketch[:-1,:2]

        R = np.random.rand()

        if R < 0.9:
            self.tr.append( sketch[1:, :] )
        elif R >= 0.9 and R < 0.95:
            self.ts.append( sketch[1:, :] )
        else:
            self.vl.append( sketch[1:, :])

    def flush(self):
        tr = np.array(self.tr, dtype=np.object)
        ts = np.array(self.ts, dtype=np.object)
        vl = np.array(self.vl, dtype=np.object)

        with open(self.filepath, 'wb') as f:
            np.savez(f, train=tr, test=ts, valid=vl)