from abc import ABC, abstractmethod
import numpy as np
from ..src.utils import setseed


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
    """

    def __init__(self, mean, kernel):
        self._mean = mean
        self._kernel = kernel

    @setseed('numpy')
    def __call__(self, size, seed=None):
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


class BlobScaler(GPSampler):
    """Gaussian Process sampler augmented with postprocessing method
    to map outputs in [0, 2] and hence read it as a scaling factor
    """

    def _as_scaling_factor(self, x):
        return np.clip(1 + x, a_min=0, a_max=2)

    def __call__(self, size, seed=None):
        X = super().__call__(size=size, seed=seed)
        X = self._as_scaling_factor(X)
        return X
