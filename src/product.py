from PIL import Image
import numpy as np
import random
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
    __mode__ = ['random', 'grid']

    def __init__(self, size, mode='random', grid_size=None, color=0, img_mode='L',
                 blob_transform=None, pdf=np.random.rand, seed=None, blobs={}):
        super(Product, self).__init__(blobs)
        self._size = size
        self._mode = mode
        self._bg = Image.new(size=size, color=color, mode=img_mode)
        self._blob_transform = blob_transform
        self._pdf = pdf
        self._seed = seed

        assert mode in Product.__mode__, f"Invalid mode, must be in {Product.__mode__}"
        if mode == 'grid':
            self._grid_size = grid_size
            self._build_grid(seed=seed)

    @setseed('random')
    def _build_grid(self, seed=None):
        """Builds grid anchors location list

        A public self.grid attribute is created, containing all available grid
            patching locations
        Private self._shuffled_grid attribute is rather used when patching to
            favor scattered patching location when nb of blobs < nb locations

        Args:
            seed (int): random seed (default: None)
        """
        # Generate (nb_anchor, 2) array with possible patching locations
        x = np.arange(0, self.size[0], self.grid_size[0])
        y = np.arange(0, self.size[1], self.grid_size[1])
        grid = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)

        # Randomly perturbate grid to avoid over-regular patterns
        eps = self.pdf(*grid.shape).astype(int)
        grid += eps

        # Record ordered gris locations as public attribute
        grid_locs = list(map(tuple, grid))
        self._grid = grid_locs[:]

        # Record shuffled grid locations as private attribute
        random.shuffle(grid_locs)
        self._shuffled_grid = iter(grid_locs)

    def add(self, blob, loc):
        """Registers blob

        Args:
            blob (Blob): blob instance to register
            loc (tuple[int]): upper-left corner if 2-tuple, upper-left and
                lower-right corners if 4-tuple
        """
        # If blob has an idx, use it
        if blob.idx:
            idx = blob.idx
        # Else create a new one
        else:
            idx = len(self)

        self[idx] = (loc, blob)
        blob.affiliate()

    @setseed('numpy')
    def random_add(self, blob, seed=None):
        """Draws random location for patching

        Args:
            blob (Blob): blob instance to register
            seed (int): random seed (default: None)
        """
        if self.mode == 'random':
            # Draw random patching location and register
            loc = self._rdm_loc(blob, seed=seed)
        elif self.mode == 'grid':
            try:
                # Choose unfilled location from grid
                loc = next(self._shuffled_grid)

            except StopIteration:
                raise IndexError("No space left on grid")
        self.add(blob, loc)

    @setseed('random')
    def _augment_blob(self, blob, seed=None):
        """If defined, applies transformation to blob
        Args:
            blob (Blob)

        Returns:
            type: Blob
        """
        # If blob defines its own transformation, use it
        if blob.aug_func:
            aug_blob = blob.augment(seed=seed)
        # Elif product defines blobs transformation, use it
        elif self.blob_transform:
            aug_blob = self.blob_transform(blob)
            aug_blob = blob._new(aug_blob.im)
        # Else use blob as is
        else:
            aug_blob = blob
        return aug_blob

    @setseed('random')
    def generate(self, seed=None):
        """Generates image of background with patched blobs

        Returns:
            type: PIL.Image.Image
        """
        # Copy background image
        img = self.bg.copy()

        for loc, blob in self.values():
            # Apply blob transformation
            blob = self._augment_blob(blob=blob)
            # Paste on background with transparency mask
            img.paste(blob, loc, mask=blob)
        return img

    @setseed('numpy')
    def _rdm_loc(self, blob, seed=None):
        x = int(self.bg.width * self.pdf())
        y = int(self.bg.height * self.pdf())
        return x, y

    @property
    def size(self):
        return self._size

    @property
    def mode(self):
        return self._mode

    @property
    def bg(self):
        return self._bg

    @property
    def blob_transform(self):
        return self._blob_transform

    @property
    def pdf(self):
        return self._pdf

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def grid(self):
        return self._grid

    @property
    def seed(self):
        return self._seed
