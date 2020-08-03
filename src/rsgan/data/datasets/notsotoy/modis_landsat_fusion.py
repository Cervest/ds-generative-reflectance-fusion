import os
from operator import add
from functools import reduce
from torch.utils.data import Dataset
from src.notsotoygeneration.preprocessing import PatchDataset
from src.rsgan.data import DATASETS


class PatchFusionDataset(PatchDataset):
    """Extends PatchDataset by returning returning last known landsat frame, current
    modis frame and current landsat frame as target instead, i.e.

        Model input : (landsat_{t-1}, modis_t)
        Model target : (landsat_t)
    """
    def __getitem__(self, idx):
        """Loads frame arrays

        Args:
            idx (int): dataset index - corresponds to time step

        Returns:
            type: tuple[np.ndarray]
        """
        modis_frame, landsat_frame = super().__getitem__(idx + 1)

        last_landsat_path = self._landsat_path[idx]
        last_landsat_frame = self._load_array(path=last_landsat_path)
        last_landsat_frame = self._apply_transform(last_landsat_frame)

        return (last_landsat_frame, modis_frame), landsat_frame

    def __len__(self):
        length = super().__len__() - 1
        return length


@DATASETS.register('modis_landsat_temporal_resolution_fusion')
class MODISLandsatTemporalResolutionFusionDataset(Dataset):
    """Class for temporal and resolutional fusion of MODIS and Landsat frames task

    Loads patches dataset from all available locations in root directory and returns
    items following PatchFusionDataset.__getitem__

    Args:
        root (str): path to directory where patches have been dumped
        transform (callable): np.ndarray -> np.ndarray optional transform for patches
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.dataset = self._load_datasets()

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
        # Load Patch datasets of each individual view
        datasets = [PatchFusionDataset(root=os.path.join(self.root, patch_directory), transform=self.transform)
                    for patch_directory in os.listdir(self.root)]

        # Concatenate into single dataset
        dataset = reduce(add, datasets)
        return dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
