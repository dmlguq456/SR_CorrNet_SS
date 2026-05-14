import numpy as np
import torch
from tensorboardX import SummaryWriter
import matplotlib.pylab as plt
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')


# https://pytorch.org/docs/stable/tensorboard.html

class MyWriter(SummaryWriter):
    def __init__(self, logdir, n_fft=512, n_hop=256, sr=16000):
        super(MyWriter, self).__init__(logdir, flush_secs=1)

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.sr = sr

        self.window = torch.hann_window(window_length=n_fft, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False)

    def log_audio(self, wav, label='label', step=0, normalize=True):
        wav = wav.detach().cpu().numpy()
        if normalize:
            wav = wav / (np.max(np.abs(wav)) + 1.0e-3)
        self.add_audio(label, wav, step, self.sr)

    def log_spec(self, data, label, step):
        self.add_image(label,
            spec_to_plot(data), step, dataformats='HWC')

    def log_mag(self, data, label, step):
        self.add_image(label,
            mag_to_plot(data), step, dataformats='HWC')

    def log_wav2spec(self, src, key, step, normalize=True):
        if normalize:
            src = src / (torch.max(torch.abs(src)) + 1.0e-3)
        src = torch.stft(src, n_fft=self.n_fft, hop_length=self.n_hop, window=self.window.to(src.device), center=True, normalized=False, onesided=True, return_complex=True)
        self.log_spec(src, key, step)


def fig_to_np(fig):
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def _array_to_plot(data, origin='lower', clim=(-80, 20), xlabel=None, ylabel=None):
    """Convert a 2D array to a numpy image array via matplotlib."""
    fig, ax = plt.subplots()
    im = plt.imshow(data, cmap=cm.jet, aspect='auto', origin=origin)
    plt.colorbar(im)
    plt.clim(*clim)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    fig.canvas.draw()
    img = fig_to_np(fig)
    plt.close(fig)
    return img

def spec_to_plot(data, normalized=True):
    data = data.detach().cpu().numpy()
    mag = np.abs(data)
    np.seterr(divide='warn')
    mag = 10 * np.log(mag)
    return _array_to_plot(mag, xlabel='Time', ylabel='Freq')

def mag_to_plot(data):
    mag = data.detach().cpu().numpy()
    mag = 10 * np.log(mag)
    return _array_to_plot(mag, xlabel='Time', ylabel='Freq')
