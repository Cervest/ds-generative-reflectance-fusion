import os
from operator import add
from functools import reduce
from src.toygeneration import ProductDataset
from src.rsgan.data import DATASETS
from .toy_dataset import ToyDataset


@DATASETS.register('toy_sar_to_optical')
class ToySARToOptical(ToyDataset):
    """Class for sar to optical dataset task on toy generated data

    Yields pair of matching (SAR, Optical) correponding to same view at same moment
    SAR images are corrupted with simulated speckle noise
    Optical images are corrupted with random cloud presence

    Args:
        root (str): path to dataset root directory containing subdirectories of
            ProductDataset
        use_annotations (bool): if True, also loads time series annotation mask
    """
    _sar_dirname = "sar"
    _optical_dirname = "optical"

    def __init__(self, root, use_annotations=False):
        super().__init__(root=root, use_annotations=use_annotations)
        buffer = self._load_datasets(root)
        self.sar_dataset = buffer[0]
        self.optical_dataset = buffer[1]

    def _load_datasets(self):
        """Loads and concatenates datasets from multiple views of optical and
        sar imagery

        torch.utils.Dataset instances can be concatenated as simply as
        ```
        concatenated_dataset = dataset_1 + dataset_2
        ```
        We hence here load datasets corresponding to each set of generated polygons -
        i.e. with a different seed - into lists and obtain the concatenated datasets
        by reducing these lists with addition operator

        Returns:
            type: tuple[ProductDataset]
        """
        # Init empy dataset lists
        sar_datasets = []
        optical_datasets = []

        # Fill with datasets from each individual views
        for seed in os.listdir(self.root):
            sar_datasets += [ProductDataset(os.path.join(self.root, seed, self._sar_dirname))]
            optical_datasets += [ProductDataset(os.path.join(self.root, seed, self._optical_dirname))]

        # Set horizon value = time series length - supposed same across all datasets
        self._set_horizon_value(optical_datasets[0])

        # Concatenate into single datasets
        concatenated_sar_dataset = reduce(add, sar_datasets)
        concatenated_optical_dataset = reduce(add, optical_datasets)
        return concatenated_sar_dataset, concatenated_optical_dataset

    def __getitem__(self, index):
        """Dataset frames retrieval method

        Args:
            index (int): index of frames to retrieve in dataset

        Returns:
            type: (torch.Tensor, torch.Tensor), torch.Tensor
                  (torch.Tensor, torch.Tensor), torch.Tensor, np.ndarray
        """
        # Load frames from respective datasets
        sar, _ = self.sar_dataset[index]
        optical, annotation = self.optical_dataset[index]

        # Transform as tensors and normalize
        sar, optical = list(map(self.transform, [sar, optical]))
        sar, optical = sar.float(), optical.float()

        # Format output
        if self.use_annotations:
            annotation = self.annotations_transform(annotation)
            output = (sar, optical), annotation
        else:
            output = sar, optical
        return output

    def __len__(self):
        datasets_lengths = set(map(len, [self.optical_dataset, self.sar_dataset]))
        assert len(datasets_lengths) == 1, "Datasets lengths mismatch"
        return datasets_lengths.pop()

    @property
    def sar_dataset(self):
        return self._sar_dataset

    @property
    def optical_dataset(self):
        return self._optical_dataset

    @property
    def target_dataset(self):
        return self.optical_dataset

    @sar_dataset.setter
    def sar_dataset(self, dataset):
        self._sar_dataset = dataset

    @optical_dataset.setter
    def optical_dataset(self, dataset):
        self._optical_dataset = dataset
