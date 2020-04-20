from PIL import Image


class Blob(Image.Image):
    """Any object, region, segment of an image basically
    This is a casting class for PIL.Image.Image

    Args:
        img (PIL.Image.Image): instance to cast

    Attributes:
        _affiliated (bool): if True, is associated to a product
    """

    def __init__(self, img):
        super(Blob, self).__init__()
        self.__dict__.update(img.__dict__)
        self._affiliated = False

    @property
    def affiliated(self):
        return self._affiliated

    @property
    def numel(self):
        return self.width * self.height

    def affiliate(self):
        self._affiliated = True


class Digit(Blob):
    """MNIST Digits blobs class

    Args:
        img (PIL.Image.Image): instance to cast
        id (int): digit index in dataset
        label (int): digit numerical value
    """

    def __init__(self, img, id=None, label=None):
        super(Digit, self).__init__(img=img)
        self._id = id
        self._label = label

    @property
    def id(self):
        return self._id

    @property
    def label(self):
        return self._label
