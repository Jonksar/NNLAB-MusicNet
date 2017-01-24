import numpy as np


class MusicnetDataGenerator:
    def __init__(self, ignore_files, batch_size=7500*3, mode='random', seed=20787957635):
        self.mode = mode

        if mode == 'random':
            self.seed = seed
        elif mode == 'sequential' or mode == 'normal':
            pass
        else:
            NotImplementedError("Use random or sequential")

    def __call__(self):
        """ This function returns a batch to"""
        if self.mode == 'random':
            X, Y = self._random_yield()
        elif self.mode == 'sequential' or self.mode == 'normal':
            X, Y = self._sequential_yield()

        return X, Y

    def _random_yield(self):
        pass

    def _sequential_yield(self):
        pass


