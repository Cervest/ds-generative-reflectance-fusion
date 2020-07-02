from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ToyDataset(Dataset, ABC):
    """General class factorizing behavior of toy datasets based on synthetic
    generated remote sensing data

    Args:
        root (str): path to dataset root directory containing subdirectories of
            ProductDataset
        use_annotations (bool): if True, also loads time series annotation mask
    """
    def __init__(self, root, use_annotations):
        self.root = root
        self.use_annotations = use_annotations
        self.frames_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=0.5,
                                                                         std=0.5)])
        self.annotations_transform = lambda x: x[:, :, 1]

    @abstractmethod
    def _load_datasets(self):
        """Loads and concatenates datasets from multiple views

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
        pass

    def _set_horizon_value(self, product_dataset):
        """Sets dataset time series horizon value from reference product dataset

        Args:
            product_dataset (ProductDataset):
        """
        self.horizon = product_dataset.index['features']['horizon']

    @abstractmethod
    def __len__(self):
        pass

    @property
    def root(self):
        return self._root

    @property
    def use_annotations(self):
        return self._use_annotations

    @property
    def frames_transform(self):
        return self._frames_transform

    @property
    def annotations_transform(self):
        return self._annotations_transform

    @property
    def horizon(self):
        return self._horizon

    @property
    @abstractmethod
    def target_dataset(self):
        pass

    @root.setter
    def root(self, root):
        self._root = root

    @use_annotations.setter
    def use_annotations(self, use_annotations):
        self._use_annotations = use_annotations

    @frames_transform.setter
    def frames_transform(self, transform):
        self._frames_transform = transform

    @annotations_transform.setter
    def annotations_transform(self, transform):
        self._annotations_transform = transform

    @horizon.setter
    def horizon(self, horizon):
        self._horizon = horizon

    @classmethod
    def build(cls, cfg):
        return cls(root=cfg['root'])
