<<<<<<< HEAD
import torch
=======
>>>>>>> 270ac908b464753e151d72b15f6fa3d1c5af11a5
import torch.nn as nn
from .backbones import ConvNet
from .modules import Conv2d
from ..models import MODELS


@MODELS.register('patchgan')
<<<<<<< HEAD
class PatchGAN(ConvNet):
=======
class PatchGan(ConvNet):
>>>>>>> 270ac908b464753e151d72b15f6fa3d1c5af11a5
    """Implementation of PatchGan discriminator proposed by Isola et al. 2017
    in "Image-to-Image Translation with Conditional Adversarial Networks"

    for real : just a CNN with sigmoid activation

    Args:
        input_size (tuple[int]): (C, H, W)
        n_filters (list[int]): list of number of filter of each
            convolutional layer
        conv_kwargs (dict, list[dict]): kwargs of convolutional layers, if dict
            same for each convolutional layer
    """
<<<<<<< HEAD
    _base_kwargs = {'kernel_size': 4, 'stride': 2, 'bn': True, 'relu': True, 'leak': 0.2}

    def __init__(self, input_size, n_filters, conv_kwargs):
        super().__init__(input_size=input_size)
=======
    def __init__(self, input_size, n_filters, conv_kwargs):
        super().__init__(self, input_size=input_size)
>>>>>>> 270ac908b464753e151d72b15f6fa3d1c5af11a5
        self._conv_kwargs = self._init_kwargs_path(conv_kwargs, n_filters)

        # Build convolutional layers
        n_filters.insert(0, self.input_size[0])
        conv_sequence = [Conv2d(in_channels=n_filters[i], out_channels=n_filters[i + 1],
                         **self._conv_kwargs[i]) for i in range(len(n_filters) - 1)]
        self.conv_layers = nn.Sequential(*conv_sequence)

        # Make sigmoid layer
        self.sigmoid = nn.Sigmoid()

<<<<<<< HEAD
    def forward(self, x, source):
        x = torch.cat([x, source], dim=1)
=======
    def forward(self, x):
>>>>>>> 270ac908b464753e151d72b15f6fa3d1c5af11a5
        x = self.conv_layers(x)
        x = self.sigmoid(x)
        return x.mean()

    @classmethod
    def build(cls, cfg):
        del cfg['name']
        return cls(**cfg)
