from scipy.fftpack import fft
import time
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

train_data = np.load(open('musicnet.npz', 'rb'))

print len(train_data.files)

# load samples from song 1788 to X and Y
X, Y = train_data['1788']

fs = 44100
fig = plt.figure(figsize=(20,5))
plt.plot(X[0: 30*fs], color=(41/255, 104/255, 168/255))
fig.axes[0].set_xlim([0, 30*fs])

# Play audio
sd.play(X[0: 30 * fs])

# Parse the labels into Yvec
stride = 512                         # 512 samples between windows
wps = fs/float(stride)               # ~86 windows/second
Yvec = np.zeros((int(30*wps),128))   # 128 distinct note labels
colors = {41 : .33, 42 : .66, 43 : 1}

for window in range(Yvec.shape[0]):
        labels = Y[window*stride]
        for label in labels:
            Yvec[window,label.data[1]] = colors[label.data[0]]


fig = plt.figure(figsize=(20,5))
plt.imshow(Yvec.T,aspect='auto',cmap='ocean_r')
plt.gca().invert_yaxis()
fig.axes[0].set_xlabel('window')
fig.axes[0].set_ylabel('note (MIDI code)')


# Spectograms of the data

window_size = 2048
Xs = np.empty([int(30 * wps), 2048])

for i in range(Xs.shape[0]):
    Xs[i] = np.abs(fft(X[i * stride : i * stride + window_size]))

fig = plt.figure(figsize=(20,7))
plt.imshow(Xs.T[0:150],aspect='auto')
plt.gca().invert_yaxis()
fig.axes[0].set_xlabel('windows (~86Hz)')
fig.axes[0].set_ylabel('frequency')
plt.show()
