import numpy as np
import pandas as pd
from functools import reduce


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


def truncate_dimensions(ts_dataset, ndim):
    """Drops columns from dataframe to fit specfied number of dimensions

    Args:
        ts_dataset (timeserie.TSDataset)
        ndim (int): if negative or greater than max number of dims, dataset
            is returned unchanged

    Returns:
        type: timeserie.TSDataset
    """
    if ndim >= 0:
        truncated_dims = ts_dataset.data.columns[:ndim]
        ts_dataset.data = ts_dataset.data[truncated_dims]
    return ts_dataset


def group_labels(ts_dataset, n_groups):
    """Processes dataset labels array of size (N, ) filled with C possible
    labels values. Splits the C label values into n_groups of - if possible -
    equally sized groups and replace with group labels the original (N, ) labels

    Args:
        ts_dataset (timeserie.TSDataset)
        n_groups (int)

    Returns:
        type: timeserie.TSDataset
    """
    # Group unique labels values
    unique_labels = np.unique(ts_dataset.labels)
    grouped_labels = np.array_split(unique_labels, n_groups)

    # Map each label value to its corresponding group index
    groups_mapping = [{label: idx for label in group} for idx, group in enumerate(grouped_labels)]
    groups_mapping = reduce(lambda a, b: {**a, **b}, groups_mapping)

    # Replace actual label values with group labels
    new_labels = 1 + np.array([groups_mapping[label] for label in ts_dataset.labels])
    ts_dataset.labels = new_labels
    return ts_dataset


def min_max_rescale(ts_dataset, amin=0, amax=1):
    """Rescales dataset time series values in [amin, amax] independently along
    each dimension/column

    Args:
        ts_dataset (timeserie.TSDataset)

    Returns:
        type: timeserie.TSDataset
    """
    # Get maximum and minimum value by dimension
    df = ts_dataset.data
    min_by_dim = [np.min([x.values for x in df[col]]) for col in df.columns]
    max_by_dim = [np.max([x.values for x in df[col]]) for col in df.columns]

    # Rescale rows while keeping them encapsulated as pd.Series
    rescale = lambda x, min, max: (amax - amin) * (x - min) / (max - min) + amin
    rescale_by_dim = lambda row: pd.Series([rescale(x, min, max) for (x, min, max) in zip(row, min_by_dim, max_by_dim)])
    ts_dataset.data = df.apply(rescale_by_dim, axis=1)
    return ts_dataset


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
