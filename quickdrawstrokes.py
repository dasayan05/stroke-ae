import os, pdb
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from quickdrawutils import unpack_drawing

class QuickDrawStrokes(Dataset):
    def __init__(self, qd_root, max_sketches=5000, dtype=np.float32, start_from_zero=False):
        super().__init__()

        # Track the parameters
        if os.path.exists(qd_root):
            self.qd_root = qd_root
            self.bin_paths = [os.path.join(self.qd_root, p) for p in os.listdir(self.qd_root)]
        self.max_sketches = max_sketches
        self.dtype = dtype
        self.start_from_zero = start_from_zero

        # The cached data
        self.cache = []
        for bin_file_path in self.bin_paths:
            n_sketch_each_cat = 0
            with open(bin_file_path, 'rb') as file:
                while True:
                    try:
                        drawing = unpack_drawing(file)['image']
                        self.cache.extend(drawing)

                        n_sketch_each_cat += 1
                        if n_sketch_each_cat >= self.max_sketches:
                            break
                    except:
                        break

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        # get a stroke
        stroke = np.array(self.cache[i], dtype=self.dtype).T
        # normalize it
        norm_factor = np.sqrt((stroke**2).sum(1)).max()
        stroke = stroke / (norm_factor + np.finfo(np.float64).eps)
        
        # pen up probabilities
        q = np.zeros((stroke.shape[0], 1), dtype=self.dtype); q[-1, 0] = 1.

        if self.start_from_zero:
            stroke -= stroke[0,:]
        
        stroke = np.hstack((stroke, q)) # concat the pen up probabilities
        
        # pdb.set_trace()
        return stroke[:-1,:-1], (stroke[1:,:-1], stroke[1:,-1])

    def collate(batch):
        lengths = torch.tensor([x.shape[0] for (x, (_, _)) in batch])
        # pdb.set_trace()
        
        padded_seq_inp = pad_sequence([torch.tensor(x) for (x, (_, _)) in batch])
        padded_seq_out = pad_sequence([torch.tensor(y) for (_, (y, _)) in batch])
        padded_seq_p = pad_sequence([torch.tensor(p) for (_, (_, p)) in batch])
        
        return pack_padded_sequence(padded_seq_inp, lengths, enforce_sorted=False), \
              (pack_padded_sequence(padded_seq_out, lengths, enforce_sorted=False), \
               pack_padded_sequence(padded_seq_p, lengths, enforce_sorted=False))

    def get_dataloader(self, batch_size, shuffle = True, pin_memory = True):
        return DataLoader(self, batch_size=batch_size,
            collate_fn=QuickDrawStrokes.collate, shuffle=shuffle, pin_memory=pin_memory)

if __name__ == '__main__':
    import sys
    qds = QuickDrawStrokes(sys.argv[1], max_sketches=2)
    qdl = qds.get_dataloader(16)

    for X, (Y, P) in qdl:
        pdb.set_trace()