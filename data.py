import numpy as np
import h5py 

f = h5py.File("cache.hdh5", 'w')

dataset = f.create_dataset('cacheName', (n_data, n_channels, ))
https://www.kaggle.com/c/state-farm-distracted-driver-detection/forums/t/20664/data-can-t-fit-in-memory
