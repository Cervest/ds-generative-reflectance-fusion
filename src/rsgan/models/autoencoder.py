import torch.nn as nn
from .backbones import ConvNet
from .modules import Conv2d, ConvTranspose2d
from ..models import MODELS


@MODELS.register('autoencoder')
class AutoEncoder(ConvNet):
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
    def __init__(self, input_size, out_channels, enc_filters, dec_filters, enc_kwargs=None,
                 dec_kwargs=None, out_kwargs=None):
        super().__init__(input_size=input_size)
        self.encoder = Encoder(input_size=input_size,
                               n_filters=enc_filters,
                               conv_kwargs=enc_kwargs)

        self.decoder = Decoder(input_size=self.encoder.output_size,
                               n_filters=dec_filters,
                               conv_kwargs=dec_kwargs)

        self.output_layer = ConvTranspose2d(in_channels=dec_filters[-1],
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            relu=True,
                                            **out_kwargs)

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        x = self.output_layer(x)
        return x

    @classmethod
    def build(cls, cfg):
        del cfg['name']
        return cls(**cfg)


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
    _base_kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'relu': True}

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
    _base_kwargs = {'kernel_size': 4, 'stride': 2, 'relu': True}

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
