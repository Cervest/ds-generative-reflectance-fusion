import torch
import torch.nn as nn
from .backbones import ConvNet, ResidualFeatureExtractor
from .modules import Conv2d, ConvTranspose2d
from ..models import MODELS


@MODELS.register('residual_autoencoder')
class ResidualAutoEncoder(ConvNet):
    """Autoencoding 2D convolutional network

    Args:
        input_size (tuple[int]): (C, H, W)
        out_channels (int): number of output channels
        enc_filters (list[int]): list of number of filter of each
            convolutional layer
        dec_filters (list[int]): list of number of filter of each
            convolutional layer
        enc_kwargs (dict, list[dict]): kwargs of encoding path, if dict same for
            each convolutional layer
        dec_kwargs (dict, list[dict]): kwargs of decoding path, if dict same for
            each convolutional layer
        out_kwargs (dict): kwargs of output layer
    """
    def __init__(self, input_size, out_channels, enc_filters, n_blocks, dec_filters,
                 in_kwargs=None, enc_kwargs=None, dec_kwargs=None, out_kwargs=None):
        super().__init__(input_size=input_size)
        in_kwargs = {} if in_kwargs is None else in_kwargs
        out_kwargs = {} if out_kwargs is None else out_kwargs

        self.input_layer = Conv2d(in_channels=input_size[0],
                                  out_channels=enc_filters[0],
                                  kernel_size=3,
                                  padding=1,
                                  stride=2,
                                  relu=False,
                                  bn=False,
                                  bias=False,
                                  **in_kwargs)

        encoder_input_size = (enc_filters[0],) + tuple(input_size[1:])
        self.encoder = ResidualFeatureExtractor(input_size=encoder_input_size,
                                                n_filters=enc_filters,
                                                n_blocks=n_blocks,
                                                conv_kwargs=enc_kwargs)

        self.decoder = Decoder(input_size=self.encoder.output_size,
                               n_filters=dec_filters,
                               conv_kwargs=dec_kwargs)

        self.output_layer = Conv2d(in_channels=dec_filters[-1],
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   **out_kwargs)

    def forward(self, x):
        x = self.input_layer(x)
        latent = self.encoder(x)
        x = self.decoder(latent)
        x = self.output_layer(x)
        return x

    @classmethod
    def build(cls, cfg):
        kwargs = cfg.copy()
        del kwargs['name']
        return cls(**kwargs)


@MODELS.register('two_stream_autoencoder')
class TwoStreamResidualAutoEncoder(ConvNet):
    """AutoEncoder with two separate encoding streams and a fusing convolutional
    layer

    Args:
        input_size (tuple[int]): (C, H, W)
        out_channels (int): number of output channels
        enc_filters (list[int]): list of number of filter of each
            convolutional layer
        dec_filters (list[int]): list of number of filter of each
            convolutional layer
        enc_kwargs (dict, list[dict]): kwargs of encoding path, if dict same for
            each convolutional layer
        dec_kwargs (dict, list[dict]): kwargs of decoding path, if dict same for
            each convolutional layer
        out_kwargs (dict): kwargs of output layer
    """
    def __init__(self, input_size, out_channels, enc_filters_1, enc_filters_2,
                 n_blocks, dec_filters, enc_kwargs_1=None, enc_kwargs_2=None,
                 dec_kwargs=None, in_kwargs=None, out_kwargs=None):
        super().__init__(input_size=input_size)
        in_kwargs = {} if in_kwargs is None else in_kwargs
        out_kwargs = {} if out_kwargs is None else out_kwargs

        self.input_layer = Conv2d(in_channels=input_size[0],
                                  out_channels=enc_filters_1[0],
                                  kernel_size=3,
                                  padding=1,
                                  stride=2,
                                  relu=False,
                                  bn=False,
                                  bias=False,
                                  **in_kwargs)
        encoder_input_size = (enc_filters_1[0],) + tuple(input_size[1:])
        self.encoder_1 = ResidualFeatureExtractor(input_size=encoder_input_size,
                                                  n_filters=enc_filters_1,
                                                  n_blocks=n_blocks,
                                                  conv_kwargs=enc_kwargs_1)

        self.encoder_2 = Encoder(input_size=input_size,
                                 n_filters=enc_filters_2,
                                 conv_kwargs=enc_kwargs_2)

        latent_size_1 = self.encoder_1.output_size
        latent_size_2 = self.encoder_2.output_size
        latent_size = (latent_size_1[0] + latent_size_2[0],) + latent_size_1[1:]

        self.fuse = Conv2d(in_channels=latent_size[0], out_channels=latent_size[0],
                           kernel_size=3, padding=1, relu=True)

        self.decoder = Decoder(input_size=latent_size,
                               n_filters=dec_filters,
                               conv_kwargs=dec_kwargs)

        self.output_layer = Conv2d(in_channels=dec_filters[-1],
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   **out_kwargs)

    def forward(self, x1, x2):
        x1 = self.input_layer(x1)
        latent_1 = self.encoder_1(x1)
        latent_2 = self.encoder_2(x2)
        fused_latent = self.fuse(torch.cat([latent_1, latent_2], dim=1))
        x = self.decoder(fused_latent)
        output = self.output_layer(x)
        return output


class Encoder(ConvNet):
    """Encoding 2D convolutional network - conv blocks use strided convolution,
    batch normalization and relu activation

    Args:
        input_size (tuple[int]): (C, H, W)
        n_filters (list[int]): list of number of filter of each convolutional
            layer
        conv_kwargs (dict, list[dict]): kwargs of decoding path, if dict same for
            each convolutional layer
    """
    _base_kwargs = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'relu': True, 'bn': True}

    def __init__(self, input_size, n_filters, conv_kwargs=None):
        super().__init__(input_size=input_size)
        self._conv_kwargs = self._init_kwargs_path(conv_kwargs, n_filters)

        # Build encoding layers
        C, H, W = self.input_size
        n_filters.insert(0, C)
        encoding_seq = [Conv2d(in_channels=n_filters[i], out_channels=n_filters[i + 1],
                        **self._conv_kwargs[i]) for i in range(len(n_filters) - 1)]
        self.encoding_layers = nn.Sequential(*encoding_seq)

    def forward(self, x):
        output = self.encoding_layers(x)
        return output


class Decoder(ConvNet):
    """Decoding 2D convolutional network - conv blocks use strided deconvolution,
    batch normalization and relu activation

    Args:
        input_size (tuple[int]): (C, H, W)
        n_filters (list[int]): list of number of filter of each convolutional
            layer
        conv_kwargs (dict, list[dict]): kwargs of decoding path, if dict same for
            each convolutional layer
    """
    _base_kwargs = {'kernel_size': 4, 'stride': 2, 'relu': True, 'bn': True, 'padding': 1}

    def __init__(self, input_size, n_filters, conv_kwargs=None):
        super().__init__(input_size=input_size)
        self._conv_kwargs = self._init_kwargs_path(conv_kwargs, n_filters)

        # Build decoding layers
        C, H, W = self.input_size
        n_filters.insert(0, C)
        decoding_seq = [ConvTranspose2d(in_channels=n_filters[i], out_channels=n_filters[i + 1],
                        **self._conv_kwargs[i]) for i in range(len(n_filters) - 1)]
        self.decoding_layers = nn.Sequential(*decoding_seq)

    def forward(self, x):
        output = self.decoding_layers(x)
        return output
