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

    Args:
        mean (callable): mean function np.ndarray -> np.ndarray
        kernel (sklearn.gaussian_process.kernel): kernel function (np.ndarray, np.ndarray) -> np.ndarray
        size (tuple[int]): optional default size for sampled vectors
    """

    def __init__(self, mean, kernel, kernel_name, size):
        self._mean = mean
        self._kernel = kernel
        self._size = size
        self._kernel_name = kernel_name
        if size:
            self._mu, self._choleskies = self._compute_params(size)

    @classmethod
    def _cache_cholesky(cls, name, size, kernel):
        global CHOLESKY
        t = cls._get_sampling_points(size)
        cholesky = cls._compute_cholesky_kronecker_decomposition(kernel, t)
        CHOLESKY[name] = cholesky

    @classmethod
    def _get_sampling_points(cls, size):
        """Instantiates tuple of 1D arrange-like representing the GP sampling points

        Args:
            size (tuple[int]): (length,) or (height, width) like tuples

        Returns:
            type: tuple[np.ndarray]
        """
        if len(size) == 1:
            x = np.arange(0, size[0])
            sampling_points = (np.expand_dims(x, -1), )
        elif len(size) == 2:
            x = np.arange(0, size[0])
            y = np.arange(0, size[1])
            sampling_points = (np.expand_dims(x, -1), np.expand_dims(y, -1))
        else:
            raise ValueError(f"Requested size has {len(size)} dimensions - up to 2 dimenions supported")
        return sampling_points

    @classmethod
    def _compute_mean(cls, mean, sampling_points):
        """Applies mean function according to dimensionality of sampling points

        Args:
            mean (callable): mean function np.ndarray -> np.ndarray
            sampling_points (tuple[np.ndarray]): tuple of sampling points arrays

        Returns:
            type: np.ndarray
        """
        if len(sampling_points) == 1:
            mu = mean(sampling_points[0])
        elif len(sampling_points) == 2:
            x, y = sampling_points
            grid = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
            mu = mean(grid)
        else:
            raise ValueError(f"Requested size has {len(sampling_points)} dimensions - up to 2 dimenions supported")
        return mu

    @classmethod
    def _compute_cholesky_kronecker_decomposition(cls, kernel, sampling_points):
        """Compute cholesky factor of kronecker components of covariance matrix
        on each individual sampling axis

        Args:
            kernel (sklearn.gaussian_process.kernel): kernel function (np.ndarray, np.ndarray) -> np.ndarray
            sampling_points (tuple[np.ndarray]): tuple of sampling points arrays

        Returns:
            type: list[np.ndarray]
        """
        choleskies = []
        for sampling_axis in sampling_points:
            K = kernel(sampling_axis)
            choleskies += [np.dual.cholesky(K)]
        return choleskies

    def _compute_params(self, size):
        """Computes mean vector and cholesky decomposition given sampling points
        size

        Args:
            size (tuple[int]): (length,) or (height, width) or
                (height, width, channels) of sampling points array
        Returns:
            type: np.ndarray, list[np.ndarray]
        """
        # Get default GP sampling positions
        t = self._get_sampling_points(size)

        # Save mean vector and cholesky factor
        mu = self._compute_mean(self.mean, t)
        cholesky = self._compute_cholesky_kronecker_decomposition(self.kernel, t)
        return mu, cholesky

    def _scale_by_cholesky(self, cholesky, x):
        """Scales N(0,I) vector by covariance sqrt matrix

        Unidimensional : Lx = L @ x
        Bidimensional : Lx = vector(L2 @ x.reshape(len(L2), len(L2)) @ L1.T)

        Args:
            cholesky (tuple[np.ndarray]): kronecker components of covariance matrix
                cholesky decomposition
            x (np.ndarray): N(0,I) sampled vector

        Returns:
            type: np.ndarray
        """
        if len(cholesky) == 1:
            Lx = cholesky[0].dot(x.T)
        elif len(cholesky) == 2:
            L1, L2 = cholesky
            Lx = x.reshape(-1, len(L2), len(L1))
            Lx = Lx.dot(L1.T)
            Lx = L2.dot(Lx.transpose((2, 1, 0)))
        else:
            raise ValueError(f"Provided {len(cholesky)} kronecker components - up to 2 supported")
        return Lx.reshape(-1, x.shape[-1])

    def _multivariate_normal(self, mu, cholesky, size=None):
        """Samples from multivariate normal distribution with specified mean
        vector and covariance matrix with cholesky decomposition of covariance

        Straight derived from : https://github.com/numpy/numpy/blob/master/numpy/random/_generator.pyx

        Args:
            mu (np.ndarray): mean vector of size (N, )
            cholesky (tuple[np.ndarray]): kronecker components of covariance matrix
                cholesky decomposition
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

        # Set output shape as sampling size + distribution dimensionality
        output_shape = list(shape[:])
        output_shape.append(mu.shape[0])

        # Sample from N(0, I) + rescale and shift
        x = np.random.standard_normal(output_shape).reshape(-1, mu.shape[0])
        x = mu + self._scale_by_cholesky(cholesky, x)
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
        size, channels = self._handle_input_dims(size)

        # Set default parameters values
        if size == self.size:
            mu, cholesky = self._mu, self._choleskies

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

    def _handle_input_dims(self, size):
        """Handles separation between width, height anf channels

        Args:
            size (tuple[int]): (length,) or (height, width) or
                (height, width, channels) of inducing points array
        Returns:
            type: tuple[int], int
        """
        channels = None
        size = size or self.size
        if not size:
            raise TypeError("Must specify a sampling size")
        if len(size) == 3:
            channels = size[-1]
            size = size[:2]
        return size, channels

    def _reshape_output(self, X, size, channels):
        """Reshape flattened output to input size and sets channels last

        Args:
            X (np.array): (channels, width*height) array
            size (tuple[int]): Description of parameter `size`.
            channels (type): Description of parameter `channels`.

        Returns:
            type: Description of returned object.

        """
        # Reshape to sampling points format
        output_shape = size
        if channels:
            output_shape = (channels, ) + output_shape
        X = X.reshape(output_shape)
        # Channels last
        if channels:
            X = X.transpose((1, 2, 0))
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
