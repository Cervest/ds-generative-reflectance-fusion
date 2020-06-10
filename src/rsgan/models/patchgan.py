import torch
import torch.nn as nn
from .backbones import ConvNet
from .modules import Conv2d
from ..models import MODELS


@MODELS.register('patchgan')
class PatchGAN(ConvNet):
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
    _base_kwargs = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bn': True, 'relu': True, 'leak': 0.2}

    def __init__(self, input_size, n_filters, conv_kwargs):
        super().__init__(input_size=input_size)
        self._conv_kwargs = self._init_kwargs_path(conv_kwargs, n_filters)

        # Extract inputs nb of channels to define first convolutional layer
        C, H, W = self.input_size
        n_filters.insert(0, C)

        # Build convolutional layers
        conv_sequence = [Conv2d(in_channels=n_filters[i], out_channels=n_filters[i + 1],
                         **self._conv_kwargs[i]) for i in range(len(n_filters) - 1)]
        self.conv_layers = nn.Sequential(*conv_sequence)

        # Make sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, source):
        """Runs forward pass on input tensor x conditionned on source tensor.

        Typically, when training GANs, input tensor x corresponds to real of fake sample
        and source to noise or conditionning tensor used for generation

        Args:
            x (torch.Tensor): input tensor
            source (torch.Tensor): conditionning tensor

        Returns:
            type: Description of returned object.
        """
        x = torch.cat([x, source], dim=1)
        x = self.conv_layers(x)
        x = self.sigmoid(x)
        return x.view(x.size(0), -1)

    @classmethod
    def build(cls, cfg):
        del cfg['name']
        return cls(**cfg)
