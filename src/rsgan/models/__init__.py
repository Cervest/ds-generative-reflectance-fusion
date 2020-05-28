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


__all__ = ['build_model', 'AutoEncoder']
