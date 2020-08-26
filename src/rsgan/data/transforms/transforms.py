import numpy as np

class RandomNumpyFlip:

    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            x = np.flipud(x)
        if np.random.rand() < self.p:
            x = np.fliplr(x)
        return x
