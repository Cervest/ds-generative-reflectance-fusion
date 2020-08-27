import torch.nn as nn
from .convnet import ConvNet
from ..modules import Conv2d, ResBlock


class ResidualFeatureExtractor(ConvNet):
    """Feature extractor backbone for residual architectures

    # TODO : add structure description

    Args:
        n_filters (list[int]): number of filter of each residual layer
        n_blocks (list[int]): number of blocks of each residual layer
        conv_kwargs (dict, list[dict]): kwargs of residual layers, if dict same for
            each convolutional layer
    """
    _base_kwargs = {'kernel_size': 4, 'stride': 2, 'scaling': 0.1}

    def __init__(self, n_filters, n_blocks, input_size=None, conv_kwargs=None):
        super().__init__(input_size=input_size)
        # Initialize kwargs of each residual block
        self._conv_kwargs = self._init_kwargs_path(conv_kwargs, n_filters)

        # Building residual layers
        residual_layers_seq = []
        for i, n_block in enumerate(n_blocks):
            # First block handles tensor dimension alteration
            residual_seq = [ResBlock(in_channels=n_filters[i], out_channels=n_filters[i + 1],
                                     **self._conv_kwargs[i])]

            # Following blocks conserve tensor dimensions
            residual_seq += [ResBlock(in_channels=n_filters[i + 1], out_channels=n_filters[i + 1])
                             for _ in range(n_block - 1)]

            residual_layers_seq += [nn.Sequential(*residual_seq)]

        # Encapsulate layers as sequential
        self.layers = nn.Sequential(*residual_layers_seq)

    def forward(self, x):
        return self.layers(x)


class ResNet(ConvNet):
    """ResNet backbone architecture including an input convolutional layer,
    a sequence of residual layers and an output convolutional layer

    Args:
        input_size (tuple[int]): (C, H, W)
        n_filters (list[int]): number of filter of each residual layer
        n_blocks (list[int]): number of blocks of each residual layer
        out_channels (int): number of channels of output
        conv_kwargs (dict, list[dict]): kwargs of residual layers, if dict same for
            each convolutional layer
        input_kwargs (dict): kwargs of input layer
        out_kwargs (dict): kwargs of output layer
    """
    _base_kwargs = {'stride': 2, 'scaling': 0.1}

    def __init__(self, input_size, n_filters, n_blocks, out_channels,
                 conv_kwargs=None, input_kwargs=None, out_kwargs=None):
        super().__init__(input_size=input_size)

        # Initialize input, output and residual layers kwargs
        input_kwargs = {} if input_kwargs is None else input_kwargs
        out_kwargs = {} if out_kwargs is None else out_kwargs

        # Input layer : downsamples input by 2
        self.input_layer = Conv2d(in_channels=input_size[0],
                                  out_channels=n_filters[0],
                                  kernel_size=7,
                                  stride=2,
                                  padding=3,
                                  relu=True,
                                  bn=True,
                                  bias=False,
                                  **input_kwargs)

        self.residual_layers = ResidualFeatureExtractor(n_filters=n_filters,
                                                        n_blocks=n_blocks,
                                                        conv_kwargs=conv_kwargs)

        self.output_layer = Conv2d(in_channels=n_filters[-1],
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   **out_kwargs)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_layers(x)
        x = self.output_layer(x)
        return x
