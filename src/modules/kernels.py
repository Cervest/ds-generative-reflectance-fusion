import numpy as np
import scipy
from functools import wraps

# TODO : factorize following functions together


def rbf(sigma):
    """Creates rbf kernel callable

    Args:
        sigma (float): standard deviation
    Returns:
        type: callable
    """
    @wraps('rbf')
    def wrapper(x1, x2):
        """Computes gram matrix of pair of vectors for rbf kernel

        Args:
            x1 (np.ndarray)
            x2 (np.ndarray)
        Returns:
            type: np.ndarray
        """
        x1 = np.expand_dims(x1, -1)
        x2 = np.expand_dims(x2, -1)
        norm = -0.5 * scipy.spatial.distance.cdist(x1, x2, 'sqeuclidean') / np.square(sigma)
        return np.exp(norm)
    return wrapper


def heat_kernel(size, sigma=1.):
    """Heat kernel generation utility
    Inspired from https://stackoverflow.com/a/43346070

    Args:
        size (tuple[int]): kernel (width, height)
        sigma (float): standard deviation

    Returns:
        type: np.ndarray
    """
    w, h = size
    x = np.linspace(-(h - 1) / 2., (h - 1) / 2., h)
    y = np.linspace(-(w - 1) / 2., (w - 1) / 2., w)
    grid = np.meshgrid(x, y)
    kernel = np.exp(-0.5 * (np.square(grid[0]) + np.square(grid[1])) / np.square(sigma))
    return kernel / kernel.sum()
