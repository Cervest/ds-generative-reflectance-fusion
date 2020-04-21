from PIL import Image
from src.utils import setseed


class Blob(Image.Image):
    """Any object, region, segment of an image basically
    This is a casting class for PIL.Image.Image

    Args:
        img (PIL.Image.Image): instance to cast

    Attributes:
        _affiliated (bool): if True, is associated to a product
    """

    def __init__(self, img, aug_func=None):
        super().__init__()
        self.set_img(img)
        self._aug_func = aug_func
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
                  'aug_func': self.aug_func}
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

    def set_img(self, img):
        self.__dict__.update(img.__dict__)

    def set_augmentation(self, aug_func):
        self._aug_func = aug_func

    @property
    def affiliated(self):
        return self._affiliated

    @property
    def numel(self):
        return self.width * self.height

    @property
    def aug_func(self):
        return self._aug_func

    def affiliate(self):
        self._affiliated = True


class Digit(Blob):
    """MNIST Digits blobs class

    Args:
        img (PIL.Image.Image): instance to cast
        idx (int): digit index in dataset
        label (int): digit numerical value
    """
    def __init__(self, img, idx=None, label=None, aug_func=None):
        super().__init__(img=img, aug_func=aug_func)
        self._idx = idx
        self._label = label

    def _new(self, im):
        new = super()._new(im)
        kwargs = {'img': new,
                  'idx': self.idx,
                  'label': self.label,
                  'aug_func': self.aug_func}
        new = self._build(**kwargs)
        return new

    @property
    def idx(self):
        return self._idx

    @property
    def label(self):
        return self._label
