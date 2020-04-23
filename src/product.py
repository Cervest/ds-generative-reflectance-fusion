import os
from PIL import Image
import numpy as np
import random
from src.utils import setseed


class Product(dict):
    """Plain product class, composed of a background image and multiple blobs

    > Registers blobs as a dictionnary {idx: (location_on_bg, blob)}
    > Proposes fully random and grid based patching strategy for blobs
    > Generates view of patched image on the fly

    - 'random' mode : each blob location is computed as the product height and
        width scaled by the specified random distribution. Better used with
        distribution valued in [0, 1]
    - 'grid' mode : a regular grid perturbed by the specifed random distribution
        is used to determine blobs locations. Better used with centered distributions

    Args:
        size (tuple[int]): (width, height) for background
        mode (str): patching strategy in {'random', 'grid'}
        grid_size (tuple[int]): grid cells dimensions as (width, height)
        color (int, tuple[int]): color value for background (0-255) according to mode
        blob_transform (callable): geometric transformation to apply blobs when patching
        rdm_dist (callable): numpy random distribution to use for randomization
        seed (int): random seed
        blobs (dict): hand made dict formatted as {idx: (location, blob)}
    """
    __mode__ = ['random', 'grid']

    def __init__(self, size, ndim=1, mode='random', grid_size=None, color=0,
                 blob_transform=None, rdm_dist=np.random.rand, seed=None, blobs={}):
        super(Product, self).__init__(blobs)
        self._size = size
        self._ndim = ndim
        self._mode = mode
        self._bg = Image.new(size=size, color=color, mode='L')
        self._blob_transform = blob_transform
        self._rdm_dist = rdm_dist
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
        eps = self.rdm_dist(*grid.shape).astype(int)
        grid += eps

        # Record ordered gris locations as public attribute
        grid_locs = list(map(tuple, grid))
        self._grid = grid_locs[:]

        # Record shuffled grid locations as private attribute
        random.shuffle(grid_locs)
        self._shuffled_grid = iter(grid_locs)

    def register(self, blob, loc, seed=None):
        """Registers blob

        Args:
            blob (Blob): blob instance to register
            loc (tuple[int]): upper-left corner if 2-tuple, upper-left and
                lower-right corners if 4-tuple
        """
        assert blob.ndim == self.ndim, f"Trying to add {blob.ndim}-dim blob while product is {self.ndim}-dim"
        # If blob has an idx, use it
        if blob.idx:
            idx = blob.idx
        # Else create a new one
        else:
            idx = len(self)

        # Apply product defined random geometric augmentation
        blob = self._augment_blob(blob=blob, seed=seed)
        self[idx] = (loc, blob)
        blob.affiliate()

    @setseed('numpy')
    def random_register(self, blob, seed=None):
        """Registers blob to product applying random strategy
        for the choice of its patching location

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
        self.register(blob, loc, seed)

    @setseed('random')
    def _augment_blob(self, blob, seed=None):
        """If defined, applies transformation to blob
        Args:
            blob (Blob)
            seed (int): random seed (default: None)
        Returns:
            type: Blob
        """
        # If product defines blobs transformation, use it
        if self.blob_transform:
            aug_blob = self.blob_transform(blob)
            aug_blob = blob._new(aug_blob.im)
        # Else use blob as is
        else:
            aug_blob = blob
        return aug_blob

    def view(self, seed=None):
        """Generates image of background with patched blobs
        Returns:
            type: PIL.Image.Image
        """
        # Copy background image
        img = self.bg.copy()
        for loc, blob in self.values():
            # Paste on background with transparency mask
            img.paste(blob, loc, mask=blob)
        return img

    def prepare(self):
        """Prepares product for generation by unfreezing blobs and setting up
        some hidden cache attributes
        iteration
        """
        for _, blob in self.values():
            blob.unfreeze()
        # Save array version of background in cache
        bg_array = np.expand_dims(self.bg, -1)
        bg_array = np.tile(bg_array, self.ndim)
        self.bg.array = bg_array

    def generate(self, output_dir):
        """
        Args:
            output_dir (type): Description of parameter `output_dir`.

        Returns:
            type: Description of returned object.
        """
        self.prepare()
        for i in range(self.horizon):
            # Copy background image
            img = self.bg.array.copy()
            for loc, blob in self.values():
                # Scale by time serie
                patch = next(blob)
                # Paste on background
                img = Product.patch_array(img, patch, loc)
            output_path = os.path.join(output_dir, f"step_{i}.npy")
            np.save(img, output_path)

    @setseed('numpy')
    def _rdm_loc(self, blob, seed=None):
        x = int(self.bg.width * self.rdm_dist())
        y = int(self.bg.height * self.rdm_dist())
        return x, y

    @staticmethod
    def patch_array(bg_array, blob_array, loc):
        x, y = loc
        w, h, _ = blob_array.shape
        bg_array[x:x + w, y:y + h] += blob_array
        return bg_array.clip(max=1)

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
    def rdm_dist(self):
        return self._rdm_dist

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def grid(self):
        return self._grid

    @property
    def ndim(self):
        return self._ndim

    @property
    def seed(self):
        return self._seed
