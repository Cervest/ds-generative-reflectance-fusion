import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.toygeneration import ProductDataset
from src.rsgan.data import DATASETS


@DATASETS.register('dummy_time_series')
class DummyTimeSeries(Dataset):

    def __init__(self, root, time_serie_dirname):
        self.time_series_datasets = self._load_dataset(root, time_serie_dirname)
        self.frame_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=0.5,
                                                                        std=0.5)])
        self.annotation_transform = transforms.ToTensor()

    def _load_dataset(self, root, time_serie_dirname):
        # Init empy dataset lists
        time_serie_datasets = []

        # Fill with datasets from each individual views
        for seed in os.listdir(root):
            time_serie_datasets += [ProductDataset(os.path.join(root, seed, time_serie_dirname))]

        # Set horizon value = time series length - supposed same across all datasets
        self.horizon = self._get_horizon_value(time_serie_datasets[0])

        return time_serie_datasets

    def _get_horizon_value(self, product_dataset):
        horizon = product_dataset.index['features']['horizon']
        return horizon

    def __getitem__(self, index):
        # Load time serie dataset
        time_series_dataset = self.time_series_datasets[index]

        # Unload frames and annotations - keep first annotation only
        frames, annotations = zip(*[time_series_dataset[i] for i in range(len(time_series_dataset))])
        annotation = annotations[0][:, :, 1]

        # Transform frames as tensors and normalize
        frames = list(map(self.frame_transform, frames))
        annotation = self.annotation_transform(annotation).squeeze()

        # Reshape frames as (height * width, horizon, channel) and flatten annotation
        frames = torch.stack(frames)
        horizon, channels, height, width = frames.shape
        frames = frames.permute(2, 3, 1, 0).reshape(-1, channels, horizon).float()
        annotation = annotation.flatten().float()

        # Filter out background pixels
        foreground = annotation != 0
        frames = frames[foreground]
        annotation = annotation[foreground] - 1
        return frames, annotation

    def __len__(self):
        return len(self.time_series_datasets)

    @classmethod
    def build(cls, cfg):
        return cls(root=cfg['root'], time_serie_dirname=cfg['time_serie_dirname'])
