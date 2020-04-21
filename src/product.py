from PIL import Image
import numpy as np
from src.utils import setseed


class Product(dict):
    """Plain product class, composed of a background image and multiple blobs

    > Registers blobs as a dictionnary {idx: (location_on_bg, blob)}
    > Generates view of patched image on the fly

    Args:
        size (tuple[int]): (width, height) for background
        color (int, tuple[int]): color value for background (0-255) according to mode
        mode (str): background and blobs image mode
        blob_transform (callable): geometric transformation to apply blobs when patching
        blobs (dict): hand made dict formatted as {idx: (location, blob)}
    """

    def __init__(self, size, color=0, mode='L', blob_transform=None, blobs={}):
        super(Product, self).__init__(blobs)
        self._size = size
        self._bg = Image.new(size=size, color=color, mode=mode)
        self._blob_transform = blob_transform

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

    @setseed('random')
    def random_add(self, blob, seed=None):
        if self.blob_transform:
            aug_blob = self.blob_transform(blob)
            aug_blob = blob._new(aug_blob.im)
        elif blob.aug_func:
            aug_blob = blob.augment(seed=seed)
        else:
            aug_blob = blob
        loc = self._rdm_loc(blob, seed=seed)
        self.add(aug_blob, loc)

    def generate(self):
        """Generates image of background with patched blobs

        Returns:
            type: PIL.Image.Image
        """
        img = self.bg.copy()
        for loc, blob in self.values():
            img.paste(blob, loc, mask=blob)
        return img

    @setseed('numpy')
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
    def blob_transform(self):
        return self._blob_transform
