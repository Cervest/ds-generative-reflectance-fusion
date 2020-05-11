import numpy as np
from .blob import Blob


class Digit(Blob):
    """MNIST Digits blobs class

    Extends blob with :

     - Additional attributes such as idx from MNIST dataset and label
     - Image systematic binarization

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
        super().__init__(img=img.point(lambda p: p > threshold and 255),
                         aug_func=aug_func,
                         time_serie=time_serie,
                         scale_sampler=scale_sampler)
        self._idx = idx
        self._label = label
        self._threshold = threshold

    def _new(self, im):
        new = super(Blob, self)._new(im)
        kwargs = {'img': new,
                  'idx': self.idx,
                  'label': self.label,
                  'threshold': self.threshold,
                  'aug_func': self.aug_func,
                  'time_serie': self.time_serie,
                  'scale_sampler': self.scale_sampler}
        new = self._build(**kwargs)
        return new

    def __next__(self):
        """Yields an updated version where the digit has been resized and
        its pixel values rescaled according to the specified scale sampler
        and time serie. Annotation mask is also computed and yielded along

        Returns:
            type: (np.ndarray, np.ndarray)
        """
        blob_patch = super(Digit, self).__next__()
        annotation_mask = self.annotation_mask_from(patch_array=blob_patch)
        return blob_patch, annotation_mask

    def annotation_mask_from(self, patch_array):
        """Builds annotation mask out of array to be patched, alledgedly
        following a __next__ call on the digit

        The mask has the same width and height as the patch and up to 2 channels :

            - 1st channel: digit pixels labeled by digit unique idx
            - 2nd channel: digit pixels labeled by time serie label

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
            type: Digit
        """
        threshold = threshold or self.threshold
        binarized_img = self.point(lambda p: p > threshold and 255)
        return binarized_img

    def set_idx(self, idx):
        self._idx = idx

    @property
    def idx(self):
        return self._idx

    @property
    def label(self):
        return self._label

    @property
    def threshold(self):
        return self._threshold
