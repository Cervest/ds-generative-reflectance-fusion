from PIL import Image
import numpy as np
from functools import wraps


def setseed(func):
    """Wrap onto any function having a random seed as argument, e.g.
    fn(*args, seed, **kwargs) to set numpy random seed
    """
    @wraps(func)
    def wrapper(*args, seed=None, **kwargs):
        if seed:
            np.random.seed(seed)
        return func(*args, seed=seed, **kwargs)
    return wrapper


class Product(dict):
    """Plain product class, composed of a background image and multiple blobs

    > Registers blobs as a dictionnary {idx: (location_on_bg, blob)}
    > Generates view of patched image on the fly

    Args:
        size (tuple[int]): (width, height) for background
        color (int, tuple[int]): color value for background (0-255) according to mode
        mode (str): background and blobs image mode
        transform (callable): geometric transformation to apply blobs when patching
        blobs (dict): hand made dict formatted as {idx: (location, blob)}
    """

    def __init__(self, size, color=0, mode='L', transform=None, blobs={}):
        super(Product, self).__init__(blobs)
        self._size = size
        self._bg = Image.new(size=size, color=color, mode=mode)
        self._transform = transform

    def add(self, blob, loc):
        """Registers blob

        Args:
            blob (Blob): blob instance to register
            loc (tuple[int]): upper-left corner if 2-tuple, upper-left and
                lower-right corners if 4-tuple
        """
        # If blob has an id, use it
        if hasattr(blob, 'id'):
            idx = blob.id
        # Else create a new one
        else:
            idx = len(self)

        self[idx] = (loc, blob)
        blob.affiliate()

    def add_random(self, blob, seed=None):
        loc = self._rdm_loc(blob, seed=seed)
        if self.tranform:
            blob = self.transform(blob, seed=seed)
        self.add(blob, loc)

    def generate(self):
        """Generates image of background with patched blobs

        Returns:
            type: PIL.Image.Image
        """
        img = self.bg.copy()
        for loc, blob in self.values():
            img.paste(blob, loc, mask=blob)
        return img

    @setseed
    def _rdm_loc(self, blob, seed=None):
        x = int(self.bg.width * np.random.rand())
        y = int(self.bg.height * np.random.rand())
        return x, y

    @property
    def size(self):
        return self._size

    @property
    def bg(self):
        return self._bg

    @property
    def transform(self):
        return self._transform
