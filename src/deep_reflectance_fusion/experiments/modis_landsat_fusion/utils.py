import torch
import numpy as np


def process_tensor_for_vis(tensor, qmin, qmax):
    """Preprocess imagery tensors to plot neat visualization

        Clip + Normalize

    Args:
        tensor (torch.Tensor)
        qmin (int, float): lower clipping quantile
        qmax (int, float): upper clipping quantile

    Returns:
        type: torch.Tensor
    """
    normalize = lambda x: (x - x.min(axis=(0, 1))) / (x.max(axis=(0, 1)) - x.min(axis=(0, 1)))
    array = tensor.cpu().numpy().transpose(2, 3, 0, 1)
    upper_bound = np.percentile(array, q=qmax, axis=(0, 1))
    lower_bound = np.percentile(array, q=qmin, axis=(0, 1))
    array = array.clip(min=lower_bound, max=upper_bound)
    array = normalize(array)
    tensor = torch.from_numpy(array).permute(2, 3, 0, 1)
    return tensor
