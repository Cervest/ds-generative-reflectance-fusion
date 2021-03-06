from src.utils import Registry
"""
Registery of common datasets
"""
DATASETS = Registry()


def build_dataset(cfg):
    model = DATASETS[cfg['name']](cfg)
    return model


################################################################################


from .datasets import MODISLandsatReflectanceFusionDataset


__all__ = ['build_dataset',
           'MODISLandsatReflectanceFusionDataset']
