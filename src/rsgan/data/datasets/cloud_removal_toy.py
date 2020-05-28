import os
from torch.utils.data import Dataset
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
    def __init__(self, raw_optical_root, raw_sar_root, enhanced_optical_root):
        self._raw_optical_dataset = ProductDataset(raw_optical_root)
        self._raw_sar_dataset = ProductDataset(raw_sar_root)
        self._enhanced_optical_dataset = ProductDataset(enhanced_optical_root)

    def __getitem__(self, index):
        raw_optical, _ = self.raw_optical_dataset[index]
        raw_sar, _ = self.raw_sar_dataset[index]
        enhanced_optical, _ = self.enhanced_optical_dataset[index]
        return (raw_optical, raw_sar), enhanced_optical

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

    @classmethod
    def build(cls, cfg):
        root = cfg['root']
        dataset_kwargs = {'raw_optical_root': os.path.join(root, "clouded_optical"),
                          'raw_sar_root': os.path.join(root, "sar"),
                          'enhanced_optical_root': os.path.join(root, "clean_optical")}
        return cls(**dataset_kwargs)
