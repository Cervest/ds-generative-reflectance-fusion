from abc import ABC, abstractmethod
import numpy as np
from src.utils import setseed


class Sampler(ABC):
    """Abstract class for samplers enforcing implementation of __call__"""
    @abstractmethod
    def __call__(self):
        pass


class GPSampler(Sampler):
    """Gaussian Process sampling class

    Args:
        mean (callable): mean function np.ndarray -> np.ndarray
        kernel (callable): kernel function (np.ndarray, np.ndarray) -> np.ndarray
        size (int): optional default size for sampled vectors
    """

    def __init__(self, mean, kernel, size=None):
        self._mean = mean
        self._kernel = kernel
        self._size = size

    @setseed('numpy')
    def __call__(self, size=None, seed=None):
        size = size or self.size
        if not size:
            raise TypeError("Must specify a sampling size")
        t = np.arange(0, size)
        sigma = self.kernel(t, t)
        mu = self.mean(t)
        X = np.random.multivariate_normal(mean=mu, cov=sigma)
        return X

    @property
    def mean(self):
        return self._mean

    @property
    def kernel(self):
        return self._kernel

    @property
    def size(self):
        return self._size


class ScalingSampler(GPSampler):
    """Gaussian Process sampler augmented with postprocessing method
    to map outputs in a positive interval and hence read it as a scaling factor

    Args:
        mean (callable): mean function np.ndarray -> np.ndarray
        kernel (callable): kernel function (np.ndarray, np.ndarray) -> np.ndarray
        size (int): optional default size for sampled vectors
    """

    def _as_scaling_factor(self, x):
        return 1 + 0.5 * np.tanh(x)

    def __call__(self, size=None, seed=None):
        X = super().__call__(size=size, seed=seed)
        X = self._as_scaling_factor(X)
        return X
