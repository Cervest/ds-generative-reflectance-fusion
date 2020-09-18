import torch
import torch.nn as nn
from .backbones import ConvNet
from .modules import Conv2d, ConvTranspose2d
from ..models import MODELS


@MODELS.register('unet')
class Unet(ConvNet):
    """2D Unet implementation

    Args:
        input_size (tuple[int]): (C, H, W)
        enc_filters (list[int]): list of number of filter of each
            convolutional layer
        dec_filters (list[int]): list of number of filter of each
            convolutional layer
        out_channels (int): number of output channels
        enc_kwargs (dict, list[dict]): kwargs of encoding path, if dict same for
            each convolutional layer
        dec_kwargs (dict, list[dict]): kwargs of decoding path, if dict same for
            each convolutional layer
        out_kwargs (dict): kwargs of output layer

    """
    def __init__(self, input_size, enc_filters, dec_filters, out_channels,
                 enc_kwargs=None, dec_kwargs=None, out_kwargs=None):
        super().__init__(input_size=input_size)
        out_kwargs = {} if out_kwargs is None else out_kwargs

        self.encoder = Encoder(input_size=input_size,
                               n_filters=enc_filters,
                               conv_kwargs=enc_kwargs)

        # self.decoder = Decoder(input_size=self.encoder.output_size,
        #                        n_filters=dec_filters,
        #                        conv_kwargs=dec_kwargs

        self.decoder = NNUpsamplingDecoder(input_size=self.encoder.output_size)

        self.output_layer = Conv2d(in_channels=dec_filters[-1],
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   **out_kwargs)

    def forward(self, x):
        latent_features = self.encoder(x)
        output = self.decoder(latent_features)
        output = self.output_layer(output)
        return output

    @classmethod
    def build(cls, cfg):
        kwargs = cfg.copy()
        del kwargs['name']
        return cls(**kwargs)


class BilinearUpsamplingDecoder(ConvNet):

    def __init__(self, input_size):
        super().__init__(input_size=input_size)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convs = nn.Sequential(Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=1, padding=1, bn=True, relu='learn', dropout=0.4),
                                   Conv2d(in_channels=2 * 1024, out_channels=512, kernel_size=3, stride=1, padding=1, bn=True, relu='learn', dropout=0.4),
                                   Conv2d(in_channels=2 * 512, out_channels=256, kernel_size=3, stride=1, padding=1, bn=True, relu='learn'),
                                   Conv2d(in_channels=2 * 256, out_channels=128, kernel_size=3, stride=1, padding=1, bn=True, relu='learn'),
                                   Conv2d(in_channels=2 * 128, out_channels=64, kernel_size=3, stride=1, padding=1, bn=True, relu='learn'),
                                   Conv2d(in_channels=2 * 64, out_channels=64, kernel_size=3, stride=1, padding=1, bn=False, relu=False))

    def forward(self, features):
        x = features.pop()
        for i, layer in enumerate(self.convs):
            if i > 0:
                x = self.upsampling(x)
            x = layer(x)
            if len(features) > 0:
                x = torch.cat([x, features.pop()], dim=1)
        return x


class NNUpsamplingDecoder(ConvNet):

    def __init__(self, input_size):
        super().__init__(input_size=input_size)
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        self.convs = nn.Sequential(Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=1, padding=1, bn=True, relu='learn', dropout=0.4),
                                   Conv2d(in_channels=2 * 1024, out_channels=512, kernel_size=3, stride=1, padding=1, bn=True, relu='learn', dropout=0.4),
                                   Conv2d(in_channels=2 * 512, out_channels=256, kernel_size=3, stride=1, padding=1, bn=True, relu='learn'),
                                   Conv2d(in_channels=2 * 256, out_channels=128, kernel_size=3, stride=1, padding=1, bn=True, relu='learn'),
                                   Conv2d(in_channels=2 * 128, out_channels=64, kernel_size=3, stride=1, padding=1, bn=True, relu='learn'),
                                   Conv2d(in_channels=2 * 64, out_channels=64, kernel_size=3, stride=1, padding=1, bn=False, relu=False))

    def forward(self, features):
        x = features.pop()
        for i, layer in enumerate(self.convs):
            if i > 0:
                x = self.upsampling(x)
            x = layer(x)
            if len(features) > 0:
                x = torch.cat([x, features.pop()], dim=1)
        return x


class Encoder(ConvNet):
    """Unet encoding 2D convolutional network - conv blocks use strided convolution,
    batch normalization and relu activation

    Returns a list of the features yielded by convolutional block

    Args:
        input_size (tuple[int]): (C, H, W)
        n_filters (list[int]): list of number of filter of each convolutional
            layer
        conv_kwargs (dict, list[dict]): kwargs of decoding path, if dict same for
            each convolutional layer
    """
    _base_kwargs = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'relu': 'learn', 'bn': True}

    def __init__(self, input_size, n_filters, conv_kwargs=None):
        super().__init__(input_size=input_size)
        self._conv_kwargs = self._init_kwargs_path(conv_kwargs, n_filters)

        # Extract inputs nb of channels to define first convolutional layer
        C, H, W = self.input_size
        n_filters.insert(0, C)

        # Build encoding layers
        encoding_seq = [Conv2d(in_channels=n_filters[i], out_channels=n_filters[i + 1],
                        **self._conv_kwargs[i]) for i in range(len(n_filters) - 1)]
        self.encoding_layers = nn.Sequential(*encoding_seq)

    def _compute_output_size(self):
        """Computes model output size

        Returns:
            type: tuple[int]
        """
        x = torch.rand(2, *self.input_size)
        with torch.no_grad():
            output = self(x)
        return output[-1].shape[1:]

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.encoding_layers):
            x = layer(x)
            features += [x]
        return features


class Decoder(ConvNet):
    """Unet decoding 2D convolutional network - conv blocks use strided deconvolution,
    batch normalization and relu activation

    Assumes skip connections stack a tensor of same dimensions

    Args:
        input_size (tuple[int]): (C, H, W)
        n_filters (list[int]): list of number of filter of each convolutional
            layer
        conv_kwargs (dict, list[dict]): kwargs of decoding path, if dict same for
            each convolutional layer
    """
    _base_kwargs = {'kernel_size': 4, 'stride': 2, 'relu': 'learn', 'bn': True, 'padding': 1}

    def __init__(self, input_size, n_filters, conv_kwargs=None):
        super().__init__(input_size=input_size)
        self._conv_kwargs = self._init_kwargs_path(conv_kwargs, n_filters)

        # Build decoding layers doubling inputs nb of filters to account for skip connections
        decoding_seq = [ConvTranspose2d(in_channels=self.input_size[0], out_channels=n_filters[0],
                        **self._conv_kwargs[0])]
        decoding_seq += [ConvTranspose2d(in_channels=2 * n_filters[i], out_channels=n_filters[i + 1],
                         **self._conv_kwargs[i + 1]) for i in range(len(n_filters) - 1)]
        self.decoding_layers = nn.Sequential(*decoding_seq)

    def forward(self, features):
        x = features.pop()
        for i, layer in enumerate(self.decoding_layers):
            x = layer(x)
            if len(features) > 0:
                x = torch.cat([x, features.pop()], dim=1)
        return x
