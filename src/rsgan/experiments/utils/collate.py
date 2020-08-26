import torch
import numpy as np

"""
Default batch formatting when using pytorch dataloading modules is done as :
[(data_1, target_1), (data_2, target_2), ... , (data_n, target_n)]

where the latter tuples are usually torch.Tensor instances or np.ndarray

The following utilities are meant to process such input and manipulate data in
order to yield the batches in a more training-compliant fashion
"""


def stack_optical_with_sar(batch):
    """Stacks raw optical and sar inputs as a single array and leaves
    clean target unchanged

    Args:
        batch (list): batch as [((raw_opt, raw_sar), clean_opt)]
    """
    data, target = zip(*batch)
    data = list(map(torch.cat, data))
    data = torch.stack(data).float()
    target = torch.stack(target).float()
    return data, target


def stack_optical_sar_and_annotations(batch):
    """Stacks raw optical and sar inputs as a single array, leaves
    clean target unchanged as well as annotation mask array

    Args:
        batch (list): batch as [((raw_opt, raw_sar), clean_opt, annotation)]
    """
    data, target, annotation = zip(*batch)
    data = list(map(torch.cat, data))
    data = torch.stack(data).float()
    target = torch.stack(target).float()
    annotation = np.stack(annotation)
    return data, target, annotation
