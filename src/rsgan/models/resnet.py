import torch.nn as nn
from .backbones import ConvNet
from .modules import Conv2d, ResBlock
from ..models import MODELS


@MODELS.register('resnet')
class ResNet(ConvNet):
    """ResNet backbone architecture including an input convolutional layer,
    a sequence of residual layers and an output convolutional layer

    Args:
        input_size (tuple[int]): (C, H, W)
        n_filters_residuals (int): number of filters of residual layers - constant across layers
        n_residuals (int): number of residual layers
        out_channels (int): number of channels of output
        residuals_kwargs (dict, list[dict]): kwargs of residual path, if dict same for
            each residual layer
    """
    _base_kwargs = {'kernel_size': 3, 'padding': 1, 'scaling': 0.1}

    def __init__(self, input_size, n_filters_residuals, n_residuals, out_channels, residuals_kwargs=None):
        super().__init__(input_size=input_size)
        n_residual_filters = n_residuals * [n_filters_residuals]
        self._residuals_kwargs = self._init_kwargs_path(residuals_kwargs, n_residual_filters)

        # Extract inputs nb of channels to define first convolutional layer
        C, H, W = self.input_size
        self.input_conv = Conv2d(in_channels=C,
                                 out_channels=n_filters_residuals,
                                 kernel_size=3, padding=1,
                                 relu=True, bn=True)

        # Build residual layers
        residual_seq = [ResBlock(n_channels=n_filters_residuals, **kwargs) for kwargs in self._residuals_kwargs]
        self.residual_layers = nn.Sequential(*residual_seq)

        # Output layer
        self.output_conv = Conv2d(in_channels=n_filters_residuals,
                                  out_channels=out_channels,
                                  kernel_size=3, padding=1)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.output_conv(x)
        return x

    @classmethod
    def build(cls, cfg):
        kwargs = cfg.copy()
        del kwargs['name']
        return cls(**kwargs)
