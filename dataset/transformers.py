import numpy as np
import librosa
import torch


class AudioRead():
    def __init__(self, sr=22050, offset=0.0, duration=None):
        self.sr = sr
        self.offset = offset
        self.duration = duration

    def __call__(self, x):
        y, _ = librosa.load(x, sr=self.sr, duration=self.duration,
                            offset=self.offset)

        return y


class Zscore():
    def __init__(self, divide_sigma=True):
        self.divide_sigma = divide_sigma

    def __call__(self, x):
        y = zscore(x, self.divide_sigma)

        return y


def zscore(x, divide_sigma=True):
    if len(x.shape) == 2:
        x -= x.mean(axis=0)
        if divide_sigma:
            x /= x.std(axis=0)
    else:
        x -= x.mean()
        if divide_sigma:
            x /= x.std()

    return x


class Spectrogram():
    def __init__(self, sr=22050, n_fft=1024, hop_size=160, n_band=128,
                 spec_type='melspec'):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_band = n_band
        self.spec_type = spec_type

    def __call__(self, x):
        S = spectrogram(x, self.sr, self.n_fft, self.hop_size, self.n_band,
                        self.spec_type)

        return S


def spectrogram(x, sr, n_fft, hop_size, n_band, spec_type='melspec'):
    if spec_type == 'stft':
        S = librosa.core.stft(y=x, n_fft=n_fft, hop_length=hop_size)
        S = np.abs(S)

    else:
        S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft,
                                           hop_length=hop_size, n_mels=n_band)
        # melspectrogram has raised np.abs(S)**power, default power=2
        # so power_to_db is directly applicable
        S = librosa.core.power_to_db(S, ref=np.max)

    return S


class MinMaxNorm():
    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        if isinstance(x, list):
            return [minmaxNorm(x_i, min_val=self.min_val, max_val=self.max_val) for x_i in x]
        else:
            return minmaxNorm(x, min_val=self.min_val, max_val=self.max_val)


def minmaxNorm(x, min_val=0, max_val=1):
    x -= np.mean(x)
    x_min = np.min(x)
    x_max = np.max(x)
    nom = x - x_min
    den = x_max - x_min

    if abs(den) > 1e-4:
            return (max_val - min_val) * (nom / den) + min_val
    else:
        return nom


class ChunkDivision():
    def __init__(self, duration=0.5, sr=22050, n_fft=1024, hop_size=160):
        self.duration = duration
        self.sr = sr
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.chunk_length_frame = int(sr * duration)
        self.chunk_length_window = self.chunk_length_frame // hop_size

    def __call__(self, x):
        num_window = x.shape[1]
        indices = np.arange(self.chunk_length_window, num_window, self.chunk_length_window)
        x_chunk = np.split(x, indices_or_sections=indices, axis=1)
        x_chunk = [x_i for x_i in x_chunk if x_i.shape[1] == self.chunk_length_window]

        return x_chunk


class TransposeNumpy():
    def __call__(self, x):
        if isinstance(x, list):
            return [transposeNumpy(x_i) for x_i in x]
        else:
            return transposeNumpy(x)


def transposeNumpy(x):
    return x.T


class ToTensor():
    def __call__(self, x):
        if isinstance(x, list):
            return [toTensor(x_i) for x_i in x]
        else:
            return toTensor(x)


def toTensor(x):
    return torch.from_numpy(x).type('torch.FloatTensor')


class LoadTensor():
    def __call__(self, x):

        return torch.load(x)
