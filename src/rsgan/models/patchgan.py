import torch.nn as nn
from .backbones import ConvNet
from .modules import Conv2d
from ..models import MODELS


@MODELS.register('patchgan')
class PatchGan(ConvNet):
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
    def __init__(self, input_size, n_filters, conv_kwargs):
        super().__init__(self, input_size=input_size)
        self._conv_kwargs = self._init_kwargs_path(conv_kwargs, n_filters)

        # Build convolutional layers
        n_filters.insert(0, self.input_size[0])
        conv_sequence = [Conv2d(in_channels=n_filters[i], out_channels=n_filters[i + 1],
                         **self._conv_kwargs[i]) for i in range(len(n_filters) - 1)]
        self.conv_layers = nn.Sequential(*conv_sequence)

        # Make sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.sigmoid(x)
        return x

    @classmethod
    def build(cls, cfg):
        del cfg['name']
        return cls(**cfg)
