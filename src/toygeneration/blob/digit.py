from .blob import Blob, BinaryBlob


class Digit(BinaryBlob):
    """MNIST Digits blobs class

    Extends binary blob with :

     - Additional attributes such as idx from MNIST dataset and label

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
                         scale_sampler=scale_sampler,
                         threshold=threshold)
        self._idx = idx
        self._label = label

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
