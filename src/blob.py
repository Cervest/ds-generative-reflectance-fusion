from PIL import Image
import numpy as np
from src.utils import setseed


class Blob(Image.Image):
    """Any object, region, segment of an image basically such that all the points
    in the blob can be considered somehow similar to each other

    This is a casting class for PIL.Image.Image . All blobs use a luminance
        color mode 'L'.

    Args:
        img (PIL.Image.Image): instance to cast
        aug_func (callable): augmentation callable, should take PIL.Image.Image
            as argument and return PIL.Image.Image
        time_serie (src.timeserie.TimeSerie): time serie used to update pixels
            values within blob

    Attributes:
        _affiliated (bool): if True, is associated to a product
    """

    def __init__(self, img, aug_func=None, time_serie=None):
        super().__init__()
        self.set_img(img)
        self._aug_func = aug_func
        self._time_serie = time_serie
        self._static = True
        self._affiliated = False

    @classmethod
    def _build(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def _new(self, im):
        """Overrides PIL.Image._new method to cast output as class member
        """
        # TODO : factorize to avoid reimplementation in child class
        new = super()._new(im)
        kwargs = {'img': new,
                  'aug_func': self.aug_func,
                  'time_serie': self.time_serie}
        new = self._build(**kwargs)
        return new

    @setseed('random')
    def augment(self, seed=None):
        """Applies blob augmentation transform

        Args:
            seed (int): random seed (default: None)

        Returns:
            type: Blob
        """
        if self.aug_func:
            aug_self = self.aug_func(self)
            return self._new(aug_self.im)
        else:
            raise TypeError("Please define an augmentation callable first")

    def freeze(self):
        """Freezes iteration over blob
        """
        self._static = True

    def unfreeze(self):
        """Allows to iterate over blob and sets up attributes anticipating
        iteration
        """
        self._static = False
        # Save array version of image in cache
        self.asarray(cache=True)
        # Initialize timeserie iterator
        self._ts_iterator = iter(self.time_serie)

    def asarray(self, cache=False):
        """Converts image as a (width, height, ndim) numpy array
        where ndim is determined by the time serie dimensionality (default is 1)

        Args:
            cache (bool): if True, saves array under self.array attribute

        Returns:
            type: np.ndarray or None
        """
        img_array = np.expand_dims(self, -1) / 255.
        img_array = np.tile(img_array, self.ndim)
        if cache:
            self._array = img_array
        else:
            return img_array

    def __next__(self):
        """Yields an updated version of the blob where pixels have been scaled
        channel-wize by the next time serie values

        Returns:
            type: np.ndarray
        """
        if self.static:
            raise TypeError(f"{self} is not an iterator, unfreeze to allow iteration")
        else:
            # Draw next (n_dim, ) vector from its multivariate time serie
            ts_slice = next(self._ts_iterator)
            # Scale its associated array channel wise
            scaled_array = self.array.copy() * ts_slice
            scaled_array = scaled_array.clip(min=0, max=1)
            return scaled_array

    def set_img(self, img):
        self.__dict__.update(img.__dict__)

    def set_augmentation(self, aug_func):
        self._aug_func = aug_func

    def set_time_serie(self, time_serie):
        self._time_serie = time_serie

    def affiliate(self):
        self._affiliated = True

    @property
    def static(self):
        return self._static

    @property
    def affiliated(self):
        return self._affiliated

    @property
    def time_serie(self):
        return self._time_serie

    @property
    def aug_func(self):
        return self._aug_func

    @property
    def array(self):
        return self._array

    @property
    def ndim(self):
        if self.time_serie:
            return self.time_serie.ndim
        else:
            return len(self.getbands())


class Digit(Blob):
    """MNIST Digits blobs class

    Args:
        img (PIL.Image.Image): instance to cast
        idx (int): digit index in dataset
        label (int): digit numerical value
    """
    def __init__(self, img, idx=None, label=None, aug_func=None, time_serie=None):
        super().__init__(img=img, aug_func=aug_func, time_serie=time_serie)
        self._idx = idx
        self._label = label

    def _new(self, im):
        new = super()._new(im)
        kwargs = {'img': new,
                  'idx': self.idx,
                  'label': self.label,
                  'aug_func': self.aug_func,
                  'time_serie': self.time_serie}
        new = self._build(**kwargs)
        return new

    @property
    def idx(self):
        return self._idx

    @property
    def label(self):
        return self._label
