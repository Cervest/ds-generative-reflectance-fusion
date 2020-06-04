import os
from operator import add
from functools import reduce
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.toygeneration import ProductDataset
from src.rsgan.data import DATASETS


@DATASETS.register('dummy_cloud_removal')
class DummyCloudRemovalDataset(Dataset):
    """Class for cloud removal task on toy generated datasets

    Yields pair of (Optical, SAR) raw images with clean Optical target

    Args:
        raw_optical_root (str): path to toy clouded optical images dataset
        raw_sar_root (str): path to toy sar images dataset
        enhanced_optical_root (str): path to toy clean optical images dataset
    """
    _raw_optical_dirname = "clouded_optical"
    _raw_sar_dirname = "sar"
    _clean_optical_dirname = "clean_optical"

    def __init__(self, root, use_annotations=False):
        buffer = self._load_datasets(root)
        self._raw_optical_dataset = buffer[0]
        self._raw_sar_dataset = buffer[1]
        self._enhanced_optical_dataset = buffer[2]
        self.use_annotations = use_annotations
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=0.5,
                                                                  std=0.5)])

    def _load_datasets(self, root):
        """Loads and concatenates datasets from multiple views of raw optical,
        raw sar and enhanced optical imagery

        Args:
            root (str)

        Returns:
            type: tuple[ProductDataset]
        """
        # Init empy dataset lists
        raw_optical_dataset = []
        raw_sar_dataset = []
        enhanced_optical_dataset = []

        # Fill with datasets from each individual views
        for seed in os.listdir(root):
            raw_optical_dataset += [ProductDataset(os.path.join(root, seed, self._raw_optical_dirname))]
            raw_sar_dataset += [ProductDataset(os.path.join(root, seed, self._raw_sar_dirname))]
            enhanced_optical_dataset += [ProductDataset(os.path.join(root, seed, self._clean_optical_dirname))]

        # Concatenate into single datasets
        raw_optical_dataset = reduce(add, raw_optical_dataset)
        raw_sar_dataset = reduce(add, raw_sar_dataset)
        enhanced_optical_dataset = reduce(add, enhanced_optical_dataset)
        return raw_optical_dataset, raw_sar_dataset, enhanced_optical_dataset

    def __getitem__(self, index):
        """Dataset frames retrieval method

        Args:
            index (int): index of frames to retrieve in dataset

        Returns:
            type: (torch.Tensor, torch.Tensor), torch.Tensor
                  (torch.Tensor, torch.Tensor), torch.Tensor, np.ndarray
        """
        # Load frames from respective datasets
        raw_optical, _ = self.raw_optical_dataset[index]
        raw_sar, _ = self.raw_sar_dataset[index]
        enhanced_optical, annotations = self.enhanced_optical_dataset[index]

        # Transform as tensors and normalize
        raw_optical, raw_sar, enhanced_optical = list(map(self.transform, [raw_optical, raw_sar, enhanced_optical]))

        # Format output
        if self.use_annotations:
            output = (raw_optical, raw_sar), enhanced_optical, annotations
        else:
            output = (raw_optical, raw_sar), enhanced_optical,
        return output

    def __len__(self):
        return len(self.raw_optical_dataset)

    @property
    def raw_optical_dataset(self):
        return self._raw_optical_dataset

    @property
    def raw_sar_dataset(self):
        return self._raw_sar_dataset

    @property
    def enhanced_optical_dataset(self):
        return self._enhanced_optical_dataset

    @property
    def use_annotations(self):
        return self._use_annotations

    @use_annotations.setter
    def use_annotations(self, use_annotations):
        self._use_annotations = use_annotations

    @classmethod
    def build(cls, cfg):
        return cls(root=cfg['root'])
