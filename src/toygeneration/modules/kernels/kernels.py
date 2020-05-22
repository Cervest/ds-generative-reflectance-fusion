import numpy as np


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
