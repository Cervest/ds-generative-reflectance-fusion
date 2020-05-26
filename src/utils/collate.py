import numpy as np
import torch

"""
Default batch formatting when using pytorch dataloading modules is done as :
[(data_1, target_1), (data_2, target_2), ... , (data_n, target_n)]

where the latter tuples are usually torch.Tensor instances.

The following utilities are meant to process such input and manipulate data in
order to yield the batches in a more training-compliant fashion
"""


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
