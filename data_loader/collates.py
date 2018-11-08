import numpy as np
import torch


class PadCollate():
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        seq_len = np.array([i[0].size()[self.dim] for i in batch])
        max_len = seq_len.max()
        sort_ind = seq_len.argsort()[::-1]

        # pad according to max_len
        padded = [pad_tensor(item[0], pad=max_len, dim=self.dim)
                  for item in batch]
        # stack all
        xs = torch.stack(padded, dim=0)
        ys = torch.stack([torch.from_numpy(i[1]) for i in batch], dim=0)
        ids = torch.from_numpy(np.array([i[2] for i in batch], dtype=int))

        mask = torch.ones(xs.size())
        for i, l in enumerate(seq_len):
            mask[i, l:, :] = 0

        seq_len = seq_len[sort_ind]
        xs = xs[tuple(sort_ind), :, :]
        ys = ys[tuple(sort_ind), :]
        ids = ids[tuple(sort_ind), ]

        return xs, ys, seq_len, ids, mask

    def __call__(self, batch):
        return self.pad_collate(batch)


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)

    if pad_size[dim] > 0:
        return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
    else:
        return vec
