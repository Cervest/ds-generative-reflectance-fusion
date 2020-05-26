import numpy as np
import torch

"""
Default batch formatting when using pytorch dataloading modules is done as :
[(data_1, target_1), (data_2, target_2), ... , (data_n, target_n)]

where the latter tuples are usually torch.Tensor instances.

The following utilities are meant to process such input and manipulate data in
order to yield the batches in a more training-compliant fashion
"""


def stack_optical_and_sar(batch):
    """Stacks raw optical and sar inputs as a single array and leaved
    clean target unchanged

    Args:
        batch (list): batch as [(raw_opt, raw_sar), clean_opt]
    """
    data, target = zip(*batch)
    data = map(np.dstack, data)
    data = torch.from_numpy(np.stack(data)).float().permute(0, 3, 1, 2)
    target = torch.from_numpy(np.stack(target)).float().permute(0, 3, 1, 2)
    return data, target
