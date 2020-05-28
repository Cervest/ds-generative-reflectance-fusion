import torch
import torch.nn as nn
from .backbones import ConvNet
from .modules import Conv2d, ConvTranspose2d
from ..models import MODELS


@MODELS.register('unet')
class Unet(ConvNet):
    """Unet 2D implementation

    Args:
        input_size (tuple[int]): (C, H, W)
        enc_filters (list[int]): list of number of filter of each
            convolutional layer
        dec_filters (list[int]): list of number of filter of each
            convolutional layer
        enc_kwargs (dict, list[dict]): kwargs of encoding path, if dict same for
            each convolutional layer
        dec_kwargs (dict, list[dict]): kwargs of decoding path, if dict same for
            each convolutional layer
    """
    def __init__(self, input_size, enc_filters, dec_filters, enc_kwargs={},
                 dec_kwargs={}):
        super().__init__(input_size=input_size)
        self.encoder = Encoder(input_size=input_size,
                               n_filters=enc_filters,
                               conv_kwargs=enc_kwargs)

        self.decoder = Decoder(input_size=self.encoder.output_size,
                               n_filters=dec_filters,
                               conv_kwargs=dec_kwargs)

    def forward(self, x):
        latent_features = self.encoder(x)
        output = self.decoder(latent_features)
        output = torch.tanh(output)
        return output

    @classmethod
    def build(cls, cfg):
        del cfg['name']
        return cls(**cfg)


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
    _base_kwargs = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'relu': True, 'bn': True}

    def __init__(self, input_size, n_filters, conv_kwargs={}):
        super().__init__(input_size=input_size)
        self._conv_kwargs = self._init_kwargs_path(conv_kwargs, n_filters)

        # Build encoding layers
        C, H, W = self.input_size
        n_filters.insert(0, C)
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
    _base_kwargs = {'kernel_size': 4, 'stride': 2, 'relu': True, 'bn': True, 'padding': 1}

    def __init__(self, input_size, n_filters, conv_kwargs={}):
        super().__init__(input_size=input_size)
        self._conv_kwargs = self._init_kwargs_path(conv_kwargs, n_filters)

        # Build decoding layers
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
