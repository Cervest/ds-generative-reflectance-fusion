import numpy as np
import torch


def list_collate(batch):
    """Collates inputs and targets as lists and can thus handle images of
    different sizes

    Args:
        batch (list): batch as described above
    """
    data, target = zip(*batch)
    return list(data), list(target)


def numpy_collate(batch):
    """Collates inputs and targets as numpy and can thus be fed to an augmenter
    object from core.dataloader.utils.transforms.Augmenter

    Args:
        batch (list): batch as described above
    """
    data, target = zip(*batch)
    return np.asarray(data), torch.stack(target, 0)
