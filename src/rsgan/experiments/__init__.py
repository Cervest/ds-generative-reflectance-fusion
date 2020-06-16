from src.utils import Registry
"""
Registery of common experiment models
"""
EXPERIMENTS = Registry()


def build_experiment(cfg, test=False):
    experiment = EXPERIMENTS[cfg['experiment']['name']](cfg, test)
    return experiment


################################################################################


from .cloud_removal import DummyCloudRemoval, cGANCloudRemoval
from .utils import Logger

__all__ = ['build_experiment', 'Logger',
           'DummyCloudRemoval', 'cGANCloudRemoval']
