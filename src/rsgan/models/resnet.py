import torch.nn as nn
from .backbones import ConvNet
from .modules import Conv2d
from ..models import MODELS


class ResBlock(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, residual_scaling,
                 stride=1, padding=0, bias=True, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=1,
                         padding=0, bias=True, dilation=1, relu=True, leak=0.,
                         dropout=0., bn=True)
        self.residual_scaling = residual_scaling

    def forward(self, x):
        residual = super().forward(x)
        residual = self.residual_scaling * x
        x = x + residual
        return x


@MODELS.register('resnet')
class ResNet(ConvNet):

    _base_kwargs = {'in_channels': 256, 'out_channels': 256, 'kernel_size': 3,
                    'padding': 1, 'residual_scaling': 0.1}

    def __init__(self, input_size, n_residuals, out_channels, residuals_kwargs=None):
        super().__init__(input_size=input_size)
        self._residuals_kwargs = self._init_kwargs_path(residuals_kwargs, n_residuals * [None])

        # Extract inputs nb of channels to define first convolutional layer
        C, H, W = self.input_size
        self.input_conv = Conv2d(in_channels=C,
                                 out_channels=self._residuals_kwargs[0]['in_channels'],
                                 kernel_size=3, padding=1,
                                 relu=True, bn=True)

        # Build residual layers
        residual_seq = [ResBlock(**kwargs) for kwargs in self._residuals_kwargs]
        self.residual_layers = nn.Sequential(*residual_seq)

        # Output layer
        self.output_conv = Conv2d(in_channels=self._residuals_kwargs[-1]['out_channels'],
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
