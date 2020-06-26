import os
from operator import add
from functools import reduce
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.toygeneration import ProductDataset
from src.rsgan.data import DATASETS


@DATASETS.register('dummy_sar_to_optical')
class DummySARToOptical(Dataset):
    """Class for sar to optical dataset task on toy generated data

    Yields pair of matching (SAR, Optical) correponding to same view at same moment
    SAR images are corrupted with simulated speckle noise
    Optical images are corrupted with random cloud presence

    Args:
        sar_root (str): path to toy sar images dataset
        optical_root (str): path to toy optical images dataset
    """
    _sar_dirname = "sar"
    _optical_dirname = "optical"

    def __init__(self, root, use_annotations=False):
        buffer = self._load_datasets(root)
        self.sar_dataset = buffer[0]
        self.optical_dataset = buffer[1]
        self.use_annotations = use_annotations
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=0.5,
                                                                  std=0.5)])

    def _load_datasets(self, root):
        """Loads and concatenates datasets from multiple views of raw optical,
        raw sar and enhanced optical imagery

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
        sar_datasets = []
        optical_datasets = []

        # Fill with datasets from each individual views
        for seed in os.listdir(root):
            sar_datasets += [ProductDataset(os.path.join(root, seed, self._sar_dirname))]
            optical_datasets += [ProductDataset(os.path.join(root, seed, self._optical_dirname))]

        # Set horizon value = time series length - supposed same across all datasets
        self.horizon = self._get_horizon_value(optical_datasets[0])

        # Concatenate into single datasets
        concatenated_sar_dataset = reduce(add, sar_datasets)
        concatenated_optical_dataset = reduce(add, optical_datasets)
        return concatenated_sar_dataset, concatenated_optical_dataset

    def _get_horizon_value(self, product_dataset):
        horizon = product_dataset.index['features']['horizon']
        return horizon

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
        optical, annotations = self.optical_dataset[index]

        # Transform as tensors and normalize
        sar, optical = list(map(self.transform, [sar, optical]))
        sar, optical = sar.float(), optical.float()

        # Format output
        if self.use_annotations:
            output = (sar, optical), annotations[:, :, 1]
        else:
            output = sar, optical
        return output

    def __len__(self):
        return len(self.sar_dataset)

    @property
    def sar_dataset(self):
        return self._sar_dataset

    @property
    def optical_dataset(self):
        return self._optical_dataset

    @property
    def use_annotations(self):
        return self._use_annotations

    @property
    def horizon(self):
        return self._horizon

    @sar_dataset.setter
    def sar_dataset(self, sar_dataset):
        self._sar_dataset = sar_dataset

    @optical_dataset.setter
    def optical_dataset(self, optical_dataset):
        self._optical_dataset = optical_dataset

    @use_annotations.setter
    def use_annotations(self, use_annotations):
        self._use_annotations = use_annotations

    @horizon.setter
    def horizon(self, horizon):
        self._horizon = horizon

    @classmethod
    def build(cls, cfg):
        return cls(root=cfg['root'])
