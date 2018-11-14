import os
import warnings
import csv
import numpy as np
from torch.utils.data import Dataset
from dataset import transformers
from torchvision import transforms


class PMEmodata(Dataset):
    def __init__(self, path_to_dataset='../pmemo_dataset/PMEmo', load_transformed=None, transform=None):
        if load_transformed:
            path_to_data = os.path.join(path_to_dataset, load_transformed)
        else:
            path_to_data = os.path.join(path_to_dataset, 'Chorus')

        path_to_static = {'val': os.path.join(path_to_dataset, 'V_static.csv'),
                          'aro': os.path.join(path_to_dataset, 'A_static.csv')}
        path_to_dynamic = None  # to-do

        sval = self._read_static(path_to_static['val'])
        saro = self._read_static(path_to_static['aro'])
        dval = None  # to-do
        daro = None  # to-do

        assert np.array_equal(sval[:, 0], saro[:, 0])
        # print(len(os.listdir(path_to_data)))
        path_to_data = [os.path.join(path_to_data, i)
                        for i in os.listdir(path_to_data)
                        if i.split('.')[0].split('-')[0] in sval[:, 0]]

        # print(len(path_to_data))
        # print(len(sval), len(saro))
        if not len(path_to_data) == len(sval) == len(saro):
            warnings.warn("Number of data doesn't match up annotations; you can ignore if chunks are used.")

        self.path_to_dataset = path_to_dataset
        self.path_to_data = path_to_data
        self.path_to_static = path_to_static
        self.path_to_dynamic = path_to_dynamic
        self.static_val = sval
        self.static_aro = saro
        self.dynamic_val = dval
        self.dynamic_aro = daro
        self.data = self._collect_data()
        self.transform = transform

    def _read_static(self, path_to_label):
        dim = 'V' if 'V' in path_to_label.split('/')[-1] else 'A'

        d = []
        with open(path_to_label, newline='') as csvfile:
            rows = csv.DictReader(csvfile)
            for row in rows:
                r = [row['musicId'],
                     row['_'.join(('mean', dim))],
                     row['_'.join(('std', dim))]]
                d.append(r)

        d = np.array(d)

        return d[d[:, 0].argsort()]

    def _collect_data(self):
        collect = []
        for i, j in zip(self.static_val, self.static_aro):
            match = [k for k in self.path_to_data
                     if k.split('/')[-1].split('.')[0].split('-')[0] == i[0] == j[0]]

            if len(match) == 1:
                collect.append((match[0], i, j))
            else:
                warnings.warn("More than one data have same id; you can ignore if chunks are used.")
                for l in match:
                    collect.append((l, i, j))

        return sorted(collect)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        dat, val, aro = x[0], x[1], x[2]
        e_mean = np.array([val[1], aro[1]]).astype(float)
        e_std = np.array([val[2], aro[2]]).astype(float)
        e = np.hstack((e_mean, e_std))

        if self.transform:
            trans_dat = self.transform(dat)
            return trans_dat, e, val[0]
        else:
            return dat, e, val[0]


if __name__ == '__main__':
    d = PMEmodata()
    print("The PMEmodata works!")
    print("\nDisplay the first three items:")
    print(d[0])
    print(d[1])
    print(d[2])
    """
    d_t = PMEmodata(transform=transforms.Compose([
        transformers.AudioRead(sr=22050),
        transformers.Zscore(divide_sigma=False),
        transformers.Spectrogram(sr=22050, n_fft=1024, hop_size=512, n_band=64,
                                 spec_type='melspec'),
        transformers.TransposeNumpy(),
        transformers.ToTensor()
    ]))

    print("\nDisplay the transformed first three items:")
    print(d_t[0], "\tSpectrogram size: {}".format(d_t[0][0].size()))
    print(d_t[1], "\tSpectrogram size: {}".format(d_t[1][0].size()))
    print(d_t[2], "\tSpectrogram size: {}".format(d_t[2][0].size()))
    """
    d_t = PMEmodata(transform=transforms.Compose([
        transformers.AudioRead(sr=22050),
        transformers.Zscore(divide_sigma=False),
        transformers.Spectrogram(sr=22050, n_fft=1024, hop_size=512, n_band=64,
                                 spec_type='melspec'),
        transformers.ChunkDivision(duration=0.5, sr=22050, n_fft=2048, hop_size=1024),
        transformers.TransposeNumpy(),
        transformers.ToTensor()
    ]))
