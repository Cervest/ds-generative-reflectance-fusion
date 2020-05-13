from abc import ABC, abstractmethod
import numbers
import numpy as np
from src.utils import setseed

CHOLESKY = {}


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

    def __init__(self, mean, kernel_name, kernel=None, size=None):
        self._mean = mean
        self._kernel = kernel
        self._size = size
        self._kernel_name = kernel_name
        if size:
            self._mu, self._cholesky = self._compute_params(size)

    @classmethod
    def _cache_cholesky(cls, name, size, kernel):
        global CHOLESKY
        # Compute convariance matrix
        t = cls._get_sampling_points(size)
        cov = kernel(t)
        # Cholesky decomposition
        cholesky = np.dual.cholesky(cov)
        # Save into global dictionnary
        CHOLESKY[name] = cholesky

    @classmethod
    def _get_sampling_points(cls, size):
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

    def _compute_params(self, size):
        # Get default GP sampling positions
        t = self._get_sampling_points(size)
        n_points = len(t)

        # Save mean vector and cholesky factor
        mu = self.mean(t)
        cholesky = CHOLESKY[self.kernel_name][:n_points, :n_points]
        return mu, cholesky

    def _multivariate_normal(self, mu, cholesky, size=None):
        """Samples from multivariate normal distribution with specified mean
        vector and covariance matrix with cholesky decomposition of covariance

        Straight derived from : https://github.com/numpy/numpy/blob/master/numpy/random/_generator.pyx

        Args:
            mu (np.ndarray): mean vector of size (N, )
            cholesky (np.ndarray): cholesky decomposition of covariance matrix of size (N, N)
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
        if (len(cholesky.shape) != 2) or (cholesky.shape[0] != cholesky.shape[1]):
            raise ValueError("cov must be 2 dimensional and square")
        if mu.shape[0] != cholesky.shape[0]:
            raise ValueError("mean and cov must have same length")

        # Set output shape as sampling size + distribution dimensionality
        output_shape = list(shape[:])
        output_shape.append(mu.shape[0])

        # Sample from N(0, I) + rescale and shift
        x = np.random.standard_normal(output_shape).reshape(-1, mu.shape[0])
        x = mu + x @ cholesky.T
        x.shape = tuple(output_shape)
        return x

    @setseed('numpy')
    def __call__(self, size=None, seed=None):
        """Samples from GP on a an arange of inducing points dimensioned according
        to size specifications

        Args:
            size (tuple[int]): (length,) or (height, width) or
                (height, width, channels) of inducing points array
            seed (int): random seed

        Returns:
            type: np.ndarray
        """
        # Make sure sampling points arange can be built
        size, channels = self._handle_dims(size)

        # Set default parameters values
        if size == self.size:
            mu, cholesky = self._mu, self._cholesky

        # update if different size specified
        else:
            mu, cholesky = self._compute_params(size)

        # Sample from multivariate normal to emulate GP on sampling points
        X = self._multivariate_normal(mu=mu,
                                      cholesky=cholesky,
                                      size=channels)

        # Reshape to sampling points format
        X = self._reshape_output(X, size, channels)
        return X

    def _handle_dims(self, size):
        channels = None
        size = size or self.size
        if not size:
            raise TypeError("Must specify a sampling size")
        if len(size) == 3:
            channels = size[-1]
            size = size[:2]
        return size, channels

    def _reshape_output(self, X, size, channels):
        # Reshape to sampling points format
        output_shape = size
        if channels:
            output_shape = (channels, ) + output_shape
        X = X.reshape(output_shape)
        # Channels last
        if channels:
            X = X.permute((1, 2, 0))
        return X

    @property
    def mean(self):
        return self._mean

    @property
    def kernel(self):
        return self._kernel

    @property
    def kernel_name(self):
        return self._kernel_name

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
