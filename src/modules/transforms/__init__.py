import torchvision.transforms as tf
import imgaug.augmenters as iaa
from .transforms import *
from src import Registry
"""
Registery of common sequential transforms
"""
TRANSFORMS = Registry()


def build_transform(cfg):
    return TRANSFORMS[cfg['name']](cfg)


@TRANSFORMS.register_fn('digit')
def build_digit_transform(cfg):
    """Tranformation applied to each digit when generating ideal latent product

        - Random affine distortion
        - Random flip
        - Random perspective effect
        - Random rescaling of digits size

    Args:
        cfg (dict): configuration dict
    """
    digit_transform = tf.Compose([tf.RandomAffine(degrees=(-90, 90),
                                                  scale=(0.5, 1),
                                                  shear=(-1, 1)),
                                  tf.RandomChoice([tf.RandomHorizontalFlip(0.5),
                                                   tf.RandomVerticalFlip(0.5)]),
                                  tf.RandomPerspective(),
                                  RandomScale(scale=(cfg['min_digit_scale'], cfg['max_digit_scale']))])
    return digit_transform


@TRANSFORMS.register_fn('cloud_and_speckle')
def build_cloud_speckle_transform(cfg):
    """Tranformation applied to ideal latent product at image-level when
        degrading product quality

        - Simulates cloud-like image occultation
        - Simulates speckle noise

    Args:
        cfg (dict): configuration dict
    """
    cloud_layer = iaa.CloudLayer(intensity_mean=1,
                                 intensity_freq_exponent=10,
                                 intensity_coarse_scale=1,
                                 alpha_min=(0.5, 0.8),
                                 alpha_multiplier=0.5,
                                 alpha_size_px_max=1,
                                 alpha_freq_exponent=1,
                                 sparsity=cfg['cloud_sparsity'],
                                 density_multiplier=cfg['cloud_density'])
    speckle_noise = iaa.SaltAndPepper(cfg['salt_pepper_proportion'])
    cloud_speckle_transform = iaa.Sequential([iaa.Sometimes(cfg['cloud_probability'], cloud_layer),
                                              speckle_noise])
    return cloud_speckle_transform


@TRANSFORMS.register_fn('tangential_scale_distortion')
def build_tangential_scale_distortion_transform(cfg):
    """Tranformation applied to ideal latent product at image-level when
        degrading product quality

        - Simulates tangential scale distortion

    Args:
        cfg (dict): configuration dict
    """
    distortion = TangentialScaleDistortion(image_size=(cfg['image_width'], cfg['image_height']),
                                           mesh_size=(cfg['mesh_columns_cells'], cfg['mesh_rows_cells']),
                                           axis=cfg['axis'],
                                           growth_rate=cfg['growth_rate'])
    return distortion