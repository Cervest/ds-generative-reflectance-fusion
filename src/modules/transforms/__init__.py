from .transforms import *
import torchvision.transforms as tf
import imgaug.augmenters as iaa

###############################################################################
"""
Registery of common sequential transforms
"""

TRANSFORMS = dict()

digit_transform = tf.Compose([tf.RandomAffine(degrees=(-90, 90),
                                              scale=(0.5, 1),
                                              shear=(-1, 1)),
                              tf.RandomChoice([tf.RandomHorizontalFlip(0.5),
                                               tf.RandomVerticalFlip(0.5)]),
                              tf.RandomPerspective(),
                              RandomScale(scale=(5, 15))])


degrade_transform = iaa.Sequential()
