import numpy as np
from functools import wraps


def conv_aggregation(kernel):
    """Convolutive-based aggregation operator

    Takes in image blocks of shape (width, height, channel, block_width, block_height, 1)
    and aggregates blocks values through convolution with specified kernel

    Args:
        kernel (np.ndarray): 2-dimensional kernel

    Returns:
        type: (width, height, channel) np.ndarray
    """
    assert kernel.ndim == 2, "Kernel must be a 2-dimensional array"
    kernel = np.expand_dims(kernel, axis=-1)

    @wraps('conv_aggregation')
    def wrapper(blocks, axis):
        output = np.tensordot(blocks, kernel, axes=(axis, (0, 1, 2)))
        return output
    return wrapper
