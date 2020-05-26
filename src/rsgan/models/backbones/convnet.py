from abc import ABC, abstractmethod
import torch


class ConvNet(torch.nn.Module, ABC):

    def __init__(self, input_size):
        """General class describing networks with convolutional layers
        Args:
            input_size (tuple[int]): (C, H, W)
        """
        super(ConvNet, self).__init__()
        self._input_size = input_size

    def _init_kwargs_path(self, conv_kwargs, nb_filters):
        """Initializes convolutional path making sure making sure it
        matches the number of filters dimensions
        Args:
            conv_kwargs (dict, list[dict]): convolutional block kwargs
            nb_filters (list[int]): number of filter of each block
        """
        if isinstance(conv_kwargs, list):
            assert len(conv_kwargs) == len(nb_filters), "Kwargs and number of filters length must match"
            return [{**self.__base_kwargs, **kwargs} for kwargs in conv_kwargs]
        elif isinstance(conv_kwargs, dict):
            return len(nb_filters) * [{**self._base_kwargs, **conv_kwargs}]
        else:
            raise TypeError("kwargs must be of type dict or list[dict]")

    def _hidden_dimension_numel(self):
        """Computes number of elements of hidden dimension
        """
        raise NotImplementedError

    def _compute_output_size(self):
        """Computes model output size

        Returns:
            type: tuple[int]
        """
        x = torch.rand(1, *self.input_size)
        with torch.no_grad():
            output = self(x)
        return output.shape[1:]

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (N, C, W, H)
        """
        pass

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        if not hasattr(self, '_output_size'):
            self._output_size = self._compute_output_size()
        return self._output_size

    @input_size.setter
    def input_size(self, input_size):
        self._input_size = input_size

    @output_size.setter
    def output_size(self, output_size):
        self._output_size = output_size
