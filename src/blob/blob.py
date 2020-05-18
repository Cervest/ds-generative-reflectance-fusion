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

    def _update_size(self):
        """Draws next size scaling factor and creates new resized version of
            blob
        Returns:
            type: Blob
        """
        if self.scale_sampler is not None:
            scale = next(self._scale_iterator)
            w, h = self.size
            new_w, new_h = int(np.floor(scale * w)), int(np.floor(scale * h))
            output = self.resize((new_w, new_h))
        else:
            output = self
        return output

    def _update_pixel_values(self, array):
        """Draws next pixel scaling vector and creates rescaled version of
            blob as array
        Returns:
            type: np.ndarray
        """
        if self.time_serie is not None:
            ts_slice = next(self._ts_iterator)
            # Scale array channel wise and clip values to [0, 1] range
            scaled_array = array * ts_slice
            scaled_array = scaled_array.clip(min=0, max=1)
            output = scaled_array
        else:
            output = self.asarray()
        return output

    def __next__(self):
        """Yields an updated version of the blob where pixels have been scaled
        channel-wize by the next time serie values

        Returns:
            type: np.ndarray
        """
        if self.static:
            raise TypeError(f"{self} is not iterable, unfreeze to allow iteration")
        else:
            # Resize blob with next scaling factor
            blob = self._update_size()
            # Rescale pixel values with next time serie values
            blob = self._update_pixel_values(blob.asarray())
            return blob

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


class BinaryBlob(Blob):
    """Extends Blob class with image binarization and associated annotation
    mask features

    Args:
        img (PIL.Image.Image): instance to cast
        aug_func (callable): augmentation callable, should take PIL.Image.Image
            as argument and return PIL.Image.Image
        time_serie (src.timeserie.TimeSerie): time serie used to update pixels
            values within blob
        scale_sampler (src.modules.ScalingSampler): samples a sequence of scaling
            factors used to iteratively update blob size
        threshold (int): binarization threshold in [0-255], pixels below are
            set to 0 and pixels above set to 255
    """

    def __init__(self, img, aug_func=None, time_serie=None, scale_sampler=None,
                 threshold=100):
        super().__init__(img=img.point(lambda p: p > threshold and 255),
                         aug_func=aug_func,
                         time_serie=time_serie,
                         scale_sampler=scale_sampler)
        self._threshold = threshold

    def _new(self, im):
        """Overrides PIL.Image._new method to cast output as class member
        """
        new = super(Blob, self)._new(im)
        kwargs = {'img': new,
                  'aug_func': self.aug_func,
                  'time_serie': self.time_serie,
                  'scale_sampler': self.scale_sampler,
                  'threshold': self.threshold}
        new = self._build(**kwargs)
        return new

    def __next__(self):
        """Yields an updated version where the blob has been resized and
        its pixel values rescaled according to the specified scale sampler
        and time serie. Annotation mask is also computed and yielded along

        Returns:
            type: (np.ndarray, np.ndarray)
        """
        blob_patch = super(BinaryBlob, self).__next__()
        annotation_mask = self.annotation_mask_from(patch_array=blob_patch)
        return blob_patch, annotation_mask

    def annotation_mask_from(self, patch_array):
        """Builds annotation mask out of array to be patched, alledgedly
        following a __next__ call on the blob

        The mask has the same width and height as the patch and up to 2 channels :

            - 1st channel: blob pixels labeled by digit unique idx
            - 2nd channel: blob pixels labeled by time serie label

        Args:
            patch_array (np.darray)

        Returns:
            type: np.darray
        """
        base_mask = (patch_array.sum(axis=-1, keepdims=True) > 0).astype(int)
        mask = self.idx * base_mask
        if self.time_serie is not None:
            ts_mask = self.time_serie.label * base_mask
            mask = np.dstack([mask, ts_mask])
        return mask

    def binarize(self, threshold=None):
        """Returns binarized version of image

        Args:
            threshold (int): binarization threshold in [0-255], pixels below are
            set to 0 and pixels above set to 255

        Returns:
            type: BinaryBlob
        """
        threshold = threshold or self.threshold
        binarized_img = self.point(lambda p: p > threshold and 255)
        return binarized_img

    @property
    def threshold(self):
        return self._threshold
