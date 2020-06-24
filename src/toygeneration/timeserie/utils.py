import numpy as np


def labels_as_int(labels):
    """Maps array of labels to array of integer labels

    Args:
        labels (np.ndarray)

    Returns:
        type: np.ndarray
    """
    _, int_labels = np.unique(labels, return_inverse=True)
    int_labels += 1
    return int_labels


def pad_to_max_length(array_list):
    """Pads 1D arrays to max length among arrays

    Args:
        list_of_arrays (list[np.ndarray])

    Returns:
        type: list[np.ndarray]
    """
    lengths = [x.shape[0] for x in array_list]
    max_length = np.max(lengths)
    pad_widths = [(0, max_length - length) for length in lengths]
    padded_outputs = [np.pad(x, pad_width=width) for x, width
                      in zip(array_list, pad_widths)]
    return padded_outputs


def discretize_over_points(stats_dist, n_points):
    """Given a continuous probability distribution - typically from scipy.stats -
    it discretizes at regularly spaced n_points positions

    Args:
        stats_dist (scipy.stats.rv_continuous)
        n_points (int): number of pdf evaluation points

    Returns:
        type: np.ndarray
    """
    t = np.linspace(stats_dist.ppf(0.01), stats_dist.ppf(0.99), n_points)
    distribution = stats_dist.pdf(t)
    return distribution / distribution.sum()


def pair_labels(labels_1, labels_2):
    """Creates bijective mapping between two sets of labels

    Args:
        labels_1 (list, np.ndarray)
        labels_2 (list, np.ndarray)

    Returns:
        type: dict
    """
    unique_labels_1 = np.unique(labels_1)
    unique_labels_2 = np.unique(labels_2)
    assert len(unique_labels_1) == len(unique_labels_2), "Number of labels don't match, pairing is not possible"
    return dict(zip(unique_labels_1, unique_labels_2))


def get_each_label_positions(labels):
    """Given array of labels, returns dictionnary mapping each label value
        to its corresponding positions in the array

    Args:
        labels (list, np.ndarray)

    Returns:
        type: dict
    """
    unique_labels = np.unique(labels)
    labels_positions = {label: np.argwhere(labels == label).flatten().tolist() for label in unique_labels}
    return labels_positions
