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
    digit_transform = tf.Compose([tf.RandomAffine(degrees=(-90, 90),
                                                  scale=(0.5, 1),
                                                  shear=(-1, 1)),
                                  tf.RandomChoice([tf.RandomHorizontalFlip(0.5),
                                                   tf.RandomVerticalFlip(0.5)]),
                                  tf.RandomPerspective(),
                                  RandomScale(scale=(cfg['min_digit_scale'], cfg['max_digit_scale']))])
    return digit_transform


@TRANSFORMS.register_fn('degrade')
def build_degrade_transform(cfg):
    cloud_layer = iaa.CloudLayer(intensity_mean=0,
                                 intensity_freq_exponent=-1,
                                 intensity_coarse_scale=10,
                                 alpha_min=(0.5, 0.8),
                                 alpha_multiplier=0.5,
                                 alpha_size_px_max=5,
                                 alpha_freq_exponent=-4,
                                 sparsity=3.,
                                 density_multiplier=cfg['cloud_density'])
    speckle_noise = iaa.SaltAndPepper(cfg['p'])
    distortion = TangentialScaleDistortion(image_size=(cfg['image_width'], cfg['image_height']),
                                           mesh_size=(cfg['mesh_width'], cfg['mesh_height']),
                                           axis=cfg['axis'],
                                           growth_rate=cfg['growth_rate'])
    degrade_transform = iaa.Sequential([iaa.Sometimes(cfg['cloud_probability'], cloud_layer),
                                        speckle_noise,
                                        distortion])
    return degrade_transform
