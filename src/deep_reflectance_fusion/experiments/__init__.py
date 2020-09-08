from src.utils import Registry
"""
Registery of common experiment models
"""
EXPERIMENTS = Registry()


def build_experiment(cfg, test=False):
    experiment = EXPERIMENTS[cfg['experiment']['name']](cfg, test)
    return experiment


################################################################################


from .modis_landsat_fusion import EarlyFusionMODISLandsat, ResidualEarlyFusionMODISLandsat, cGANFusionMODISLandsat, ResidualcGANFusionMODISLandsat, DiscriminatorPerceptualLossFusionMODISLandsat
from .utils import Logger

__all__ = ['build_experiment', 'Logger',
           'EarlyFusionMODISLandsat', 'ResidualEarlyFusionMODISLandsat',
           'cGANFusionMODISLandsat', 'ResidualcGANFusionMODISLandsat',
           'DiscriminatorPerceptualLossFusionMODISLandsat']
