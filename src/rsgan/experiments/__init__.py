from src.utils import Registry
"""
Registery of common experiment models
"""
EXPERIMENTS = Registry()


def build_experiment(cfg, test=False):
    experiment = EXPERIMENTS[cfg['experiment']['name']](cfg, test)
    return experiment


################################################################################


from .dummy_cloud_removal import DummyCloudRemoval
from .cgan_cloud_removal import cGANCloudRemoval
from .utils import Logger

__all__ = ['build_experiment', 'Logger',
           'DummyCloudRemoval', 'cGANCloudRemoval']
