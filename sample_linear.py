# Basic tools & plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

# Sound device
import sounddevice as sd

# Used for label extraction
from intervaltree import Interval, IntervalTree

# Used for NN building
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential

# Since training data is too large to hold in memory, we have a custom made generator to loop over training samples
from data import MusicnetDataGenerator

fs = 44100            # samples/second
window_size = 2048    # fourier window size
d = 150              # number of features
m = 128               # number of distinct notes
stride = 512          # samples between windows
wps = fs/float(512)   # windows/second
n = 1000              # training data points per recording

print("Loading musicnet.npz file")
data = np.load(open('musicnet.npz', 'rb'))

# Split data into train, test sets.
test_data = ['2303', '2382', '1819']
train_data = [f for f in data.files if f not in test_data]

# Creating TEST set
print("Generating test data")
Xtest = np.empty([len(test_data) * 7500, d])
Ytest = np.zeros([len(test_data) * 7500, m])


for i in range(len(test_data)):
    X, Y = data[test_data[i]]
    for j in range(7500):
        s = fs + j * stride # ??? "Start from one second to give us some wiggle room for larger segments"
        Xtest[7500 * i + j] = np.abs(fft(X[s: s+window_size]))[0: d]

        # Label stuff that is in the center of the window
        for label in Y[s + d/2]:
            Ytest[7500*i + j, label.data[1]] = 1

print("Test data load & extraction successful")

print("Building neural network for classification")



