from abc import ABC, abstractmethod
import numbers
from functools import wraps
import numpy as np
from src.utils import setseed


def cache(key):
    """Disclaimer : hacky way to manage cache but haven't found anything better
    yet to handle mutable types

    Workaround decorator to allow caching method calls outputs under
    cache dictionnary attribute.

    Cached values are mapped to a key string and if existing are returned by
        default. Cached value can be overwritten by specifying cache=True

    Args:
        key (str): string key for cache dictionnary mapping
    """
    def cached_fn(fn):
        @wraps('fn')
        def wrapper(self, *args, cache=False, **kwargs):
            # Is this cache key already used ?
            is_cached = key in self._cache.keys()

            # Compute and cache result if asked to or if not cached yet
            if cache or not is_cached:
                output = fn(self, *args, cache, **kwargs)
                self._cache.update({key: output})

            # Else, cache it
            elif is_cached:
                output = self._cache[key]

            # Else, straightforward uncached computation
            else:
                output = fn(self, *args, cache, **kwargs)
            return output
        return wrapper
    return cached_fn


class Sampler(ABC):
    """Abstract class for samplers enforcing implementation of __call__"""
    @abstractmethod
    def __call__(self):
        pass


class GPSampler(Sampler):
    """Gaussian Process sampling class

    TO BE REVISITED LATER ON --> right now Cholesky still too slow for practical use

    Args:
        mean (callable): mean function np.ndarray -> np.ndarray
        kernel (sklearn.gaussian_process.kernel): kernel function (np.ndarray, np.ndarray) -> np.ndarray
        size (tuple[int]): optional default size for sampled vectors
    """

    def __init__(self, mean, kernel, size=None):
        self._mean = mean
        self._kernel = kernel
        self._size = size
        self._cache = {}

    @cache('sampling_points')
    def _get_sampling_points(self, size, cache=False):
        """Creates 1D or 2D arrange-like representing the GP sampling points

        Args:
            size (tuple[int]): (length,) or (height, width) like tuples

        Returns:
            type: np.ndarray
        """
        if len(size) == 1:
            sampling_points = np.expand_dims(np.arange(0, size[0]), -1)
        elif len(size) == 2:
            x = np.arange(0, size[0])
            y = np.arange(0, size[1])
            sampling_points = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
        else:
            raise ValueError(f"Requested size has {len(size)} dimensions - up to 2 dimenions supported")
        return sampling_points

    @cache('mu')
    def _compute_mean(self, sampling_points, cache=False):
        """Applies mean function on sampling points

        Args:
            sampling_points (np.ndarray)
            cache (bool): if True, overwrites function cached output

        Returns:
            type: np.ndarray
        """
        mu = self.mean(sampling_points)
        return mu

    @cache('cov')
    def _compute_covariance(self, sampling_points, cache=False):
        """Applies kernel function on sampling points

        Args:
            sampling_points (np.ndarray)
            cache (bool): if True, overwrites function cached output

        Returns:
            type: np.ndarray
        """
        cov = self.kernel(sampling_points).astype(np.double)
        return cov

    @cache('cholesky')
    def _cholesky_decomposition(self, cov, cache=False):
        """Performs cholesky decomposition of pd matrix

        Args:
            cov (np.ndarray): pd covariance matrix
            cache (bool): if True, overwrites function cached output

        Returns:
            type: np.ndarray
        """
        L = np.dual.cholesky(cov)
        return L

    def _multivariate_normal(self, mu, cov, size=None):
        """Samples from multivariate normal distribution with specified mean
        vector and covariance matrix with cholesky decomposition of covariance

        Straight derived from : https://github.com/numpy/numpy/blob/master/numpy/random/_generator.pyx

        Args:
            mu (np.ndarray): mean vector of size (N, )
            cov (np.ndarray): covariance matrix of size (N, N)
            size (int, tuple[int]): optional sampling size argument, default None

        Returns:
            type: np.ndarray
        """
        # Handle size format
        if size is None:
            shape = []
        elif isinstance(size, (numbers.Number, np.integer)):
            shape = [size]
        else:
            shape = size

        # Assert mean and covariance dimensions match
        if len(mu.shape) != 1:
            raise ValueError("mean must be 1 dimensional")
        if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
            raise ValueError("cov must be 2 dimensional and square")
        if mu.shape[0] != cov.shape[0]:
            raise ValueError("mean and cov must have same length")

        # Set output shape as sampling size + distribution dimensionality
        output_shape = list(shape[:])
        output_shape.append(mu.shape[0])

        # Sample from N(0, I) + rescale and shift
        x = np.random.standard_normal(output_shape).reshape(-1, mu.shape[0])
        L = self._cholesky_decomposition(cov)
        x = mu + x @ L.T
        x.shape = tuple(output_shape)
        return x

    @setseed('numpy')
    def __call__(self, size=None, n_samples=None, seed=None):
        """Samples from GP on a an arange of inducing points dimensioned according
        to size specifications

        Args:
            size (tuple[int]): (length,) or (height, width) of inducing points array
            n_samples (int): number of samplings to perform
            seed (int): random seed

        Returns:
            type: np.ndarray
        """
        # Make sure sampling points arange can be built
        size = size or self.size
        if not size:
            raise TypeError("Must specify a sampling size")

        # Generate sampling points array
        t = self._get_sampling_points(size)

        # Compute mean vector and covariance matrix
        mu = self._compute_mean(t)
        cov = self._compute_covariance(t)

        # Sample from multivariate normal to emulate GP on sampling points
        X = self._multivariate_normal(mu=mu, cov=cov, size=n_samples)

        # Reshape to sampling points format
        X = self._reshape_output(X, size, n_samples)
        return X

    def _reshape_output(self, X, size, n_samples):
        # Reshape to sampling points format
        output_shape = size
        if n_samples:
            output_shape = (n_samples, ) + output_shape
        X = X.reshape(output_shape)
        # Set n_samples as channels
        if n_samples:
            X = X.permute((1, 2, 0))
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
        kernel (callable): kernel function (np.ndarray, np.ndarray) -> np.ndarray
        size (int): optional default size for sampled vectors
    """
    def __init__(self, kernel, size=None):
        super().__init__(mean=np.zeros_like, kernel=kernel, size=size)

    def _as_scaling_factor(self, x):
        return 1 + 0.5 * np.tanh(x)

    def __call__(self, size=None, seed=None):
        X = super().__call__(size=size, seed=seed)
        X = self._as_scaling_factor(X)
        return X
