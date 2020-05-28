from src.utils import Registry
"""
Registery of common experiment models
"""
EXPERIMENTS = Registry()


def build_experiment(cfg):
    experiment = EXPERIMENTS[cfg['experiment']['name']](cfg)
    return experiment


################################################################################


from .dummy_cloud_removal import DummyCloudRemoval
from .utils import Logger

__all__ = ['build_experiment', 'DummyCloudRemoval', 'Logger']
