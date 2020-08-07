from src.utils import Registry
"""
Registery of common experiment models
"""
MODELS = Registry()


def build_model(cfg):
    model = MODELS[cfg['name']](cfg)
    return model


################################################################################

from .autoencoder import AutoEncoder
from .residual_autoencoder import ResidualAutoEncoder
from .unet import Unet, ConditionalUnet
from .patchgan import PatchGAN

__all__ = ['build_model', 'AutoEncoder', 'ResidualAutoEncoder', 'Unet', 'PatchGAN', 'ConditionalUnet']
