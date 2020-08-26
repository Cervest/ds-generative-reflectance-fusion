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
    def __init__(self, input_size, enc_filters, dec_filters, out_channels,
                 enc_kwargs=None, dec_kwargs=None, out_kwargs=None):
        super().__init__(input_size=input_size)
        out_kwargs = {} if out_kwargs is None else out_kwargs

        self.encoder = Encoder(input_size=input_size,
                               n_filters=enc_filters,
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
        latent_features = self.encoder(x)
        output = self.decoder(latent_features)
        output = self.output_layer(output)
        return output

    @classmethod
    def build(cls, cfg):
        kwargs = cfg.copy()
        del kwargs['name']
        return cls(**kwargs)


@MODELS.register('conditional_unet')
class ConditionalUnet(ConvNet):
    def __init__(self, input_size, input_size_bis, out_channels, enc_filters, enc_filters_bis,
                 dec_filters, enc_kwargs=None, enc_kwargs_bis=None, dec_kwargs=None, out_kwargs=None):
        super().__init__(input_size=input_size)
        out_kwargs = {} if out_kwargs is None else out_kwargs

        # Build main and secondary encoders
        self.encoder = Encoder(input_size=input_size,
                               n_filters=enc_filters,
                               conv_kwargs=enc_kwargs)

        self.encoder_bis = Encoder(input_size=input_size_bis,
                                   n_filters=enc_filters_bis,
                                   conv_kwargs=enc_kwargs_bis)

        # Compute concatenated latent tensors size
        latent_size = self.encoder.output_size
        latent_size_bis = self.encoder_bis.output_size
        fused_latent_size = (latent_size[0] + latent_size_bis[0],) + latent_size[1:]

        # Build latent representation fusing layer
        self.fuse = Conv2d(in_channels=fused_latent_size[0], out_channels=fused_latent_size[0],
                           kernel_size=3, padding=1, relu=True)

        # Build decoder and output layer
        self.decoder = Decoder(input_size=fused_latent_size,
                               n_filters=dec_filters,
                               conv_kwargs=dec_kwargs)
        self.output_layer = Conv2d(in_channels=dec_filters[-1],
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   **out_kwargs)

    def forward(self, x, x_bis):
        # Compute latent representations
        latent_features = self.encoder(x)
        latent_features_bis = self.encoder_bis(x_bis)

        # Fuse last feature with last feature from secondary stream
        stacked_latent = torch.cat([latent_features[-1], latent_features_bis[-1]], dim=1)
        fused_latent = self.fuse(stacked_latent)
        latent_features[-1] = fused_latent

        # Decode latent features
        decoded_latent = self.decoder(latent_features)
        output = self.output_layer(decoded_latent)
        return output

    @classmethod
    def build(cls, cfg):
        kwargs = cfg.copy()
        del kwargs['name']
        return cls(**kwargs)


@MODELS.register('dual_stream_unet')
class DualStreamUnet(ConvNet):
    class DualDecoder(ConvNet):
        _base_kwargs = {'kernel_size': 4, 'stride': 2, 'relu': True, 'bn': True, 'padding': 1}

        def __init__(self, input_size, n_filters, conv_kwargs=None):
            super().__init__(input_size=input_size)
            self._conv_kwargs = self._init_kwargs_path(conv_kwargs, n_filters)

            # Build decoding layers doubling inputs nb of filters to account for skip connections
            decoding_seq = [ConvTranspose2d(in_channels=self.input_size[0], out_channels=n_filters[0],
                            **self._conv_kwargs[0])]
            decoding_seq += [ConvTranspose2d(in_channels=3 * n_filters[i], out_channels=n_filters[i + 1],
                             **self._conv_kwargs[i + 1]) for i in range(len(n_filters) - 1)]
            self.decoding_layers = nn.Sequential(*decoding_seq)

        def forward(self, features):
            x = features.pop()
            for i, layer in enumerate(self.decoding_layers):
                x = layer(x)
                if len(features) > 0:
                    x = torch.cat([x, features.pop()], dim=1)
            return x

    def __init__(self, input_size, input_size_bis, out_channels, enc_filters, enc_filters_bis,
                 dec_filters, enc_kwargs=None, enc_kwargs_bis=None, dec_kwargs=None, out_kwargs=None):
        super().__init__(input_size=input_size)
        out_kwargs = {} if out_kwargs is None else out_kwargs

        # Build main and secondary encoders
        self.encoder = Encoder(input_size=input_size,
                               n_filters=enc_filters,
                               conv_kwargs=enc_kwargs)

        self.encoder_bis = Encoder(input_size=input_size_bis,
                                   n_filters=enc_filters_bis,
                                   conv_kwargs=enc_kwargs_bis)

        # Compute concatenated latent tensors size
        latent_size = self.encoder.output_size
        latent_size_bis = self.encoder_bis.output_size
        stacked_latent_size = (latent_size[0] + latent_size_bis[0],) + latent_size[1:]

        # Build decoder and output layer
        self.decoder = DualStreamUnet.DualDecoder(input_size=stacked_latent_size,
                                                  n_filters=dec_filters,
                                                  conv_kwargs=dec_kwargs)
        self.output_layer = Conv2d(in_channels=dec_filters[-1],
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   **out_kwargs)

    def forward(self, x, x_bis):
        # Compute latent representations
        latent_features = self.encoder(x)
        latent_features_bis = self.encoder_bis(x_bis)

        # Stack features all together
        stacked_latents = [torch.cat([f, f_bis], dim=1) for (f, f_bis) in zip(latent_features, latent_features_bis)]

        # Decode latent features
        decoded_latent = self.decoder(stacked_latents)
        output = self.output_layer(decoded_latent)
        return output

    @classmethod
    def build(cls, cfg):
        kwargs = cfg.copy()
        del kwargs['name']
        return cls(**kwargs)


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
    _base_kwargs = {'kernel_size': 4, 'stride': 2, 'relu': True, 'bn': True, 'padding': 1}

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
