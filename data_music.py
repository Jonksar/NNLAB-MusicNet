from scipy.fftpack import fft
import time
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

train_data = np.load(open('musicnet.npz', 'rb'))

print len(train_data.files)

# load samples from song 1788 to X and Y
X, Y = train_data[random.choice(train_data.keys())]


def parse_data(X, Y, stride=2048, window_size=512, n_frequency=200, plot=False):
    # samples per second
    fs = 44100

    # Parse the labels into Yvec
    stride = 2048                         # samples between windows
    Yvec = np.zeros((int(X.shape[0] / stride), 128))   # 128 distinct note labels

    # from intervaltree to "onehot" matrix
    for window in range(Yvec.shape[0]):
            labels = Y[window*stride] 
            for label in labels:
                    Yvec[window, label.data[1]] = 1


    # Spectograms of the data
    # Initialize the empty Xs
    Xs = np.empty([X.shape[0] / stride - window_size, n_frequency]) # max size: stride * (n_chunks + window_size) = samples; solve for n_chunks

    for i in range(Xs.shape[0]):
        # Do fft computation and then take the energy of that frequency
        fft_temp = fft(X[i * stride : i * stride + window_size])
        Xs[i] = np.abs(fft_temp[:n_frequency])

    if plot:
        plt.imshow(Xs.T, cmap="Greys", interpolation="nearest")
        plt.show()
        plt.imshow(Yvec[:Xs.shape[0], :].T, cmap="Greys", interpolation="nearest")
        plt.show()

    return Xs, Yvec[:Xs.shape[0], :]


xf, y = parse_data(X, Y, plot=True)
print xf.shape, y.shape
