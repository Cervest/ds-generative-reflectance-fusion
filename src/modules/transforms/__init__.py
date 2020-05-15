import torchvision.transforms as tf
import imgaug.augmenters as iaa
from .transforms import *
from src.utils import Registry
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


@TRANSFORMS.register_fn('cloud')
def build_cloud_transform(cfg):
    """Tranformation applied to ideal latent product at image-level when
        degrading product quality

        - Simulates cloud-like image occultation

    Args:
        cfg (dict): configuration dict
    """
    cloud_kwargs = {'intensity_mean': 100,                             # mean clouds color ; 0 - 255
                    'intensity_freq_exponent': -2.,                    # exponent of frequency of the intensity noise ; recommend [-2.5, -1.5]
                    'intensity_coarse_scale': 3.,                      # std of distribution used to add more localized intensity to the mean intensity
                    'alpha_min': 0.0,                                  # minimum alpha when blending cloud noise with the image
                    'alpha_multiplier': 0.2,                           # high values will lead to denser clouds wherever they are visible ; recommend [0.3, 1.0]
                    'alpha_size_px_max': cfg['sampling_scale'],        # image size at which the alpha mask is sampled, lower = larger clouds
                    'alpha_freq_exponent': -1.5,                       # exponent of frequency of the alpha mask noise, lower = coarser ; recommend [-4.0, -1.5]
                    'sparsity': cfg['cloud_sparsity'],                 # exponent ; lower = coarser ; around 1.
                    'density_multiplier': cfg['cloud_density']}        # higher = denser ; [0.5, 1.5]

    cloud_layer = iaa.CloudLayer(**cloud_kwargs)
    cloud_transform = iaa.Sometimes(cfg['cloud_probability'], cloud_layer)
    return cloud_transform


@TRANSFORMS.register_fn('speckle')
def build_speckle_transform(cfg):
    """Tranformation applied to downsampled latent product at image-level when
        degrading product quality

        - Simulates speckle noise

    Args:
        cfg (dict): configuration dict
    """
    speckle_noise = iaa.SaltAndPepper(cfg['salt_pepper_proportion'])
    return speckle_noise


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
