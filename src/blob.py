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
        scale_sampler (src.modules.ScalingSampler): samples a sequence of scaling
            factors used to iteratively update blob size

    Attributes:
        _affiliated (bool): if True, is associated to a product
    """

    def __init__(self, img, aug_func=None, time_serie=None, scale_sampler=None):
        super().__init__()
        self.set_img(img)
        self._aug_func = aug_func
        self._time_serie = time_serie
        self._scale_sampler = scale_sampler
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
                  'time_serie': self.time_serie,
                  'scale_sampler': self.scale_sampler}
        new = self._build(**kwargs)
        return new

    @setseed('random')
    def augment(self, inplace=False, seed=None):
        """Applies blob augmentation transform

        Args:
            inplace (bool): if True, modifies self instead of returning new
                instance
            seed (int): random seed (default: None)

        Returns:
            type: Blob
        """
        if self.aug_func:
            aug_self = self.aug_func(self)
            augmented_blob = self._new(aug_self.im)
            if inplace:
                self = augmented_blob
            else:
                return augmented_blob
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
        # Initialize timeserie
        if self.time_serie is not None:
            self._ts_iterator = iter(self.time_serie)
        # Initialize scales sampler
        if self.scale_sampler is not None:
            if self.time_serie is not None:
                horizon = self.time_serie.horizon
            else:
                horizon = self.scale_sampler.size
            self._scale_iterator = iter(self.scale_sampler(size=horizon))

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

    def _update_img_size(self):
        """Draws next size scaling factor and creates new resized version of
            blob

        Returns:
            type: Blob
        """
        scale = next(self._scale_iterator)
        w, h = self.size
        new_w, new_h = int(np.floor(scale * w)), int(np.floor(scale * h))
        return self.resize((new_w, new_h))

    def _update_pixel_values(self, array=None):
        """Draws next pixel scaling vector and creates rescaled version of
            blob as array

        Args:
            array (np.ndarray)
        Returns:
            type: np.ndarray
        """
        if array is None:
            array = self.array
        # Draw next (n_dim,) vector from its multivariate time serie
        ts_slice = next(self._ts_iterator)
        # Scale its associated array channel wise
        scaled_array = array * ts_slice
        scaled_array = scaled_array.clip(min=0, max=1)
        return scaled_array

    def __next__(self):
        """Yields an updated version of the blob where pixels have been scaled
        channel-wize by the next time serie values

        Returns:
            type: np.ndarray
        """
        if self.static:
            raise TypeError(f"{self} is not iterable, unfreeze to allow iteration")
        else:
            buffer = None
            if self.scale_sampler is not None:
                buffer = self._update_img_size()
            if self.time_serie is not None:
                buffer = buffer or self
                buffer = self._update_pixel_values(buffer.asarray())
            return buffer

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
    def scale_sampler(self):
        return self._scale_sampler

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
        threshold (int): binarization threshold in [0-255], pixels below are
            set to 0 and pixels above set to 255
        aug_func (callable): augmentation callable, should take PIL.Image.Image
            as argument and return PIL.Image.Image
        time_serie (src.timeserie.TimeSerie): time serie used to update pixels
            values within blob
        scale_sampler (src.modules.ScalingSampler): samples a sequence of scaling
            factors used to iteratively update blob size
    """
    def __init__(self, img, idx=None, label=None, threshold=100, aug_func=None,
                 time_serie=None, scale_sampler=None):
        super().__init__(img=img,
                         aug_func=aug_func,
                         time_serie=time_serie,
                         scale_sampler=scale_sampler)
        self._idx = idx
        self._label = label
        self._threshold = threshold

    def _new(self, im):
        new = super()._new(im)
        kwargs = {'img': new,
                  'idx': self.idx,
                  'label': self.label,
                  'threshold': self.threshold,
                  'aug_func': self.aug_func,
                  'time_serie': self.time_serie,
                  'scale_sampler': self.scale_sampler}
        new = self._build(**kwargs)
        return new

    def binarize(self, threshold=None):
        threshold = threshold or self.threshold
        binarized_img = self.point(lambda p: p > threshold and 255)
        return binarized_img

    @property
    def idx(self):
        return self._idx

    @property
    def label(self):
        return self._label

    @property
    def threshold(self):
        return self._threshold
