import os
from operator import add
from functools import reduce
from src.toygeneration import ProductDataset
from src.rsgan.data import DATASETS
from .toy_dataset import ToyDataset


@DATASETS.register('toy_cloud_removal')
class ToyCloudRemovalDataset(ToyDataset):
    """Class for cloud removal task on toy generated datasets

    Yields pair of (Optical, SAR) raw images with clean Optical target

    Args:
        root (str): path to dataset root directory containing subdirectories of
            ProductDataset
        use_annotations (bool): if True, also loads time series annotation mask
    """
    _clouded_optical_dirname = "clouded_optical"
    _sar_dirname = "sar"
    _clean_optical_dirname = "clean_optical"

    def __init__(self, root, use_annotations=False):
        super().__init__(root=root, use_annotations=use_annotations)
        buffer = self._load_datasets()
        self.clouded_optical_dataset = buffer[0]
        self.sar_dataset = buffer[1]
        self.clean_optical_dataset = buffer[2]

    def _load_datasets(self):
        """Loads and concatenates datasets from multiple views of clouded optical,
        sar and clean optical imagery

        torch.utils.Dataset instances can be concatenated as simply as
        ```
        concatenated_dataset = dataset_1 + dataset_2
        ```
        We hence here load datasets corresponding to each set of generated polygons -
        i.e. with a different seed - into lists and obtain the concatenated datasets
        by reducing these lists with addition operator

        Args:
            root (str)

        Returns:
            type: tuple[ProductDataset]
        """
        # Init empy dataset lists
        clouded_optical_datasets = []
        sar_datasets = []
        clean_optical_datasets = []

        # Fill with datasets from each individual views
        for seed in os.listdir(self.root):
            clouded_optical_datasets += [ProductDataset(os.path.join(self.root, seed, self._clouded_optical_dirname))]
            sar_datasets += [ProductDataset(os.path.join(self.root, seed, self._sar_dirname))]
            clean_optical_datasets += [ProductDataset(os.path.join(self.root, seed, self._clean_optical_dirname))]

        # Set horizon value = time series length - supposed same across all datasets
        self._set_horizon_value(clean_optical_datasets[0])

        # Concatenate into single datasets
        clouded_optical_dataset = reduce(add, clouded_optical_datasets)
        sar_dataset = reduce(add, sar_datasets)
        clean_optical_dataset = reduce(add, clean_optical_datasets)
        return clouded_optical_dataset, sar_dataset, clean_optical_dataset

    def __getitem__(self, index):
        """Dataset frames retrieval method

        Args:
            index (int): index of frames to retrieve in dataset

        Returns:
            type: (torch.Tensor, torch.Tensor), torch.Tensor
                  (torch.Tensor, torch.Tensor), torch.Tensor, np.ndarray
        """
        # Load frames from respective datasets
        clouded_optical, _ = self.clouded_optical_dataset[index]
        sar, _ = self.sar_dataset[index]
        clean_optical, annotation = self.clean_optical_dataset[index]

        # Transform as tensors and normalize
        frames_triplet = [clouded_optical, sar, clean_optical]
        frames_triplet = map(self.frames_transform, frames_triplet)
        clouded_optical, sar, clean_optical = list(map(lambda x: x.float(), frames_triplet))

        # Format output
        if self.use_annotations:
            annotation = self.annotations_transform(annotation)
            output = (clouded_optical, sar), clean_optical, annotation
        else:
            output = (clouded_optical, sar), clean_optical
        return output

    def __len__(self):
        datasets_triplet = [self.clouded_optical_dataset, self.sar_dataset, self.clean_optical_dataset]
        datasets_lengths = set(map(len, datasets_triplet))
        assert len(datasets_lengths) == 1, "Datasets lengths mismatch"
        return datasets_lengths.pop()

    @property
    def clouded_optical_dataset(self):
        return self._clouded_optical_dataset

    @property
    def sar_dataset(self):
        return self._sar_dataset

    @property
    def clean_optical_dataset(self):
        return self._clean_optical_dataset

    @property
    def target_dataset(self):
        return self.clean_optical_dataset

    @clouded_optical_dataset.setter
    def clouded_optical_dataset(self, dataset):
        self._clouded_optical_dataset = dataset

    @sar_dataset.setter
    def sar_dataset(self, dataset):
        self._sar_dataset = dataset

    @clean_optical_dataset.setter
    def clean_optical_dataset(self, dataset):
        self._clean_optical_dataset = dataset
