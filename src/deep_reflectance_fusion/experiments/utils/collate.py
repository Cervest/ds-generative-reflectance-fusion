import torch

"""
Default batch formatting when using pytorch dataloading modules is done as :
[(data_1, target_1), (data_2, target_2), ... , (data_n, target_n)]

where the latter tuples are usually torch.Tensor instances or np.ndarray

The following utilities are meant to process such input and manipulate data in
order to yield the batches in a more training-compliant fashion
"""


def stack_input_frames(batch):
    """Stacks inputs as a single array and leaves target unchanged

    Args:
        batch (list): batch as [((frame_1, frame_2), targets)]
    """
    data, target = zip(*batch)
    data = list(map(torch.cat, data))
    data = torch.stack(data).float()
    target = torch.stack(target).float()
    return data, target
