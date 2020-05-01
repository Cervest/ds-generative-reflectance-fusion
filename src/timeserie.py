import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from src.utils import setseed


class TSDataset(Dataset):
    """Time Series dataset

    Args:
        root (str): path to .ts file to load

    Attributes:
        data (pd.DataFrame): (n_sample, n_dim) dataframe where columns
            corresponds to a time serie dimension and rows to a time serie.
            Time series along one dimension are stored in each cell as pd.Serie
        labels (np.ndarray): (n_sample, ) array with each serie label
    """
    def __init__(self, root):
        self._root = root
        self._data, self._labels = load_from_tsfile_to_dataframe(root)

    def __getitem__(self, idx, t=None):
        """Series access method

        Given an index, loads serie as a (n_steps, n_dim) array along with its
            label.
        If time step t is additionally specified, label is not returned and only
            (n_dim, ) array at specified time step is returned

        Args:
            idx (int): index on time serie to access in self.data
            t (int): time step

        Returns:
            type: (np.ndarray, np.ndarray) or np.ndarray
        """
        X = np.stack([x.values for x in self.data.iloc[idx]], axis=1)
        if t:
            return X[t]
        else:
            y = self.labels[idx]
            return X, y

    def __repr__(self):
        self.data.info()
        return ""

    def __len__(self):
        return len(self.data)

    def plot(self, idx, figsize=(10, 6)):
        """Quick utility to visualize a time serie from dataframe

        Args:
            idx (int): index on time serie to access in self.data
            figsize (tuple[int]): figure size
        """
        ts, label = self[idx]
        length, n_dim = ts.shape

        fig, ax = plt.subplots(n_dim, 1, figsize=figsize)
        for i in range(n_dim):
            ax[i].plot(range(ts.shape[0]), ts[:, i], label=f"Label : {label}")
        plt.legend()
        plt.show()

    @setseed('random')
    def choice(self, seed=None, replace=True):
        """Mimics random.choice by returning random sample from dataset

        Args:
            seed (int): random seed
            replace (bool): if True, allows to pick same sample multiple times
        """
        if replace:
            idx = random.randint(0, len(self) - 1)
        else:
            if not hasattr(self, '_left_to_draw'):
                self._left_to_draw = list(range(len(self)))
                random.shuffle(self._left_to_draw)
            if len(self._left_to_draw) > 0:
                idx = self._left_to_draw.pop()
            else:
                raise IndexError("All samples have already been drawn once")
        return self[idx]

    @property
    def root(self):
        return self._root

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels


class TimeSerie:
    """Time serie iterating class

    Args:
        ts (np.ndarray): (n_steps, n_dim) array
        label (int, str): the time serie label
        horizon (int): finite horizon for the time serie (default: None)
        seed (int): random seed
    """

    def __init__(self, ts, label, horizon=None, seed=None):
        self._ts = ts
        self._label = int(float(label))
        self._horizon = horizon
        self._ndim = ts.shape[1]
        self._seed = seed

    @setseed('numpy')
    def _pick_starting_point(self, seed=None):
        """Picks a starting point for the serie
        If no horizon provided, default is 0
        Otherwise, random starting point is drawn

        Args:
            seed (int): random seed

        Returns:
            type: int
        """
        if self.horizon:
            t_start = np.random.randint(0, len(self.ts) - len(self))
        else:
            t_start = 0
        return t_start

    def __iter__(self):
        """Iterates over time serie values
        Each __next__ call yields a (n_dim, ) np.ndarray
        """
        t_start = self._pick_starting_point(seed=self.seed)
        truncated_ts = self.ts[t_start:t_start + len(self)]
        return iter(truncated_ts)

    def __len__(self):
        """Time serie length or Horizon if specified

        Returns:
            type: int
        """
        return self.horizon or len(self.ts)

    @property
    def ts(self):
        return self._ts

    @property
    def label(self):
        return self._label

    @property
    def horizon(self):
        return self._horizon

    @property
    def ndim(self):
        return self._ndim

    @property
    def seed(self):
        return self._seed
