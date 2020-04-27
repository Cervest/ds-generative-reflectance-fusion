import os
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset
from progress.bar import Bar
from src.utils import setseed, mkdir, save_json


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
        horizon (int): number of time steps to generate
        nbands (int): number of bands of the product
        mode (str): patching strategy in {'random', 'grid'}
        grid_size (tuple[int]): grid cells dimensions as (width, height)
        color (int, tuple[int]): color value for background (0-255) according to mode
        blob_transform (callable): geometric transformation to apply blobs when patching
        rdm_dist (callable): numpy random distribution to use for randomization
        seed (int): random seed
        blobs (dict): hand made dict formatted as {idx: (location, blob)}
    """
    __mode__ = ['random', 'grid']
    DUMPDIR = 'data/'
    INDEX = 'index.json'

    def __init__(self, size, horizon=None, nbands=1, mode='random', grid_size=None,
                 color=0, blob_transform=None, rdm_dist=np.random.rand,
                 seed=None, blobs={}):
        super(Product, self).__init__(blobs)
        self._size = size
        self._nbands = nbands
        self._horizon = horizon
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

        # Record ordered grid locations as public attribute
        grid_locs = list(map(tuple, grid))
        self._grid = grid_locs[:]

        # Record shuffled grid locations as private attribute
        random.shuffle(grid_locs)
        self._shuffled_grid = iter(grid_locs)

    def _assert_compatible(self, blob):
        """Ensure blob is compatible with product verifying it has :
            - Same number of bands / dimensionality
            - Same or greater horizon
        Args:
            blob (Blob)
        """
        # Verify matching number of bands
        assert blob.ndim == self.nbands, f"""Trying to add {blob.ndim}-dim blob
            while product is {self.nbands}-dim"""

        # Verify time serie horizon at least equals produc horizon
        if hasattr(blob, 'time_serie'):
            assert blob.time_serie.horizon >= self.horizon, \
                f"""Blob has {blob.time_serie.horizon} horizon while product has
                 a {self.horizon} horizon"""
        return True

    def register(self, blob, loc, seed=None):
        """Registers blob

        Args:
            blob (Blob): blob instance to register
            loc (tuple[int]): upper-left corner if 2-tuple, upper-left and
                lower-right corners if 4-tuple
        """
        # Ensure blob dimensionality and horizon match product's
        self._assert_compatible(blob)

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
            loc = self._rdm_loc(seed=seed)
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

    def view(self):
        """Generates grayscale image of background with patched blobs
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
        bg_array = np.tile(bg_array, self.nbands).astype(np.float64)
        self.bg.array = bg_array

    def generate(self, output_dir, astype='npy'):
        """Runs generation as two for loops :
        ```
        for time_step in horizon:
            for blob in registered_blobs:
                Scale blob with its next time serie slice
                Patch blob on background
            Save resulting image
        ```
        Args:
            output_dir (str): path to output directory
            astype (str): in {'npy', 'jpg'}
        """
        # Build output directory, prepare product and index dict
        self._setup_output_dir(output_dir)
        self.prepare()
        index = self._init_generation_index()
        bar = Bar("Generation", max=self.horizon)

        for i in range(self.horizon):
            # Copy background image
            img = self.bg.array.copy()
            for loc, blob in self.values():
                # Scale by time serie
                patch = next(blob)
                # Paste on background
                Product.patch_array(img, patch, loc)

            filename = '.'.join([f"step_{i}", astype])

            # Record in index
            index['files'][i] = filename
            index['features']['nframes'] += 1

            # Dump file
            output_path = os.path.join(output_dir, Product.DUMPDIR, filename)
            self.dump_array(img, output_path, astype)
            bar.next()

        # Save index
        index_path = os.path.join(output_dir, Product.INDEX)
        save_json(path=index_path, jsonFile=index)

    @setseed('numpy')
    def _rdm_loc(self, seed=None):
        """Draws random location based on product background dimensions

        Args:
            seed (int): random seed
        """
        x = int(self.bg.width * self.rdm_dist())
        y = int(self.bg.height * self.rdm_dist())
        return x, y

    @staticmethod
    def patch_array(bg_array, patch_array, loc):
        """Inplace patching of numpy array into another numpy array
        Patch is cropped if needed to handle out-of-boundaries patching

        Args:
            bg_array (np.ndarray): background array, valued in [0, 1]
            patch_array (np.ndarray): array to patch, valued in [0, 1]
            loc (tuple[int]): patching location

        Returns:
            type: None
        """
        x, y = loc
        # Crop patch if out-of-bounds upper-left patching location
        if x < 0:
            patch_array = patch_array[-x:]
            x = 0
        if y < 0:
            patch_array = patch_array[:, -y:]
            y = 0
        w, h, _ = patch_array.shape

        # Again crop if out-of-bounds lower-right patching location
        w = min(w, bg_array.shape[0] - x)
        h = min(h, bg_array.shape[1] - y)

        # Patch and clip
        bg_array[x:x + w, y:y + h] += patch_array[:w, :h]
        bg_array.clip(max=1)

    def _setup_output_dir(self, output_dir, overwrite=False):
        """Builds output directory hierarchy structured as :

            directory_name/
            └── data

        Args:
            output_dir (str): path to output directory
            overwrite (bool): if True and directory already exists, erases
                everything and recreates from scratch
        """
        mkdir(output_dir, overwrite=overwrite)
        data_dir = os.path.join(output_dir, Product.DUMPDIR)
        mkdir(data_dir)

    def _init_generation_index(self):
        """Initializes generation index

        Returns:
            type: dict
        """
        index = {'features': {'width': self.size[0],
                              'height': self.size[1],
                              'nbands': self.nbands,
                              'horizon': self.horizon,
                              'nblob': len(self),
                              'nframes': 0},
                 'files': dict()}
        return index

    def dump_array(self, array, dump_path, astype):
        """Dumps numpy array at specified location in .npy format
        Handles png format for 3-bands products only

        TODO : dirty string manipulations in here, to be refactored when
            settled on export format

        Args:
            array (np.ndarray): array to dump
            dump_path (str): output file path
            astype (str): in {'npy', 'jpg'}
        """
        if astype == 'npy':
            with open(dump_path, 'wb') as f:
                np.save(f, array)
        elif astype == 'jpg':
            assert self.nbands == 3, "RGB image generation only available for 3-bands products"
            img = Image.fromarray((array * 255).astype(np.uint8), mode='RGB')
            img.save(dump_path)
        else:
            raise TypeError("Unknown dumping type")

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
    def nbands(self):
        return self._nbands

    @property
    def horizon(self):
        return self._horizon

    @property
    def seed(self):
        return self._seed


class ProductLoader(Dataset):
    """Dataset loading class for generated products

    Very straigthforward implementation to be adapted to product dumping
        format

    Args:
        root (str): path to directory where product has been dumped
    """

    def __init__(self, root):
        self._root = root
        filenames = os.listdir(root)
        self._files_path = [os.path.join(root, file) for file in filenames]

    def __getitem__(self, idx):
        path = self._files_path[idx]
        return np.load(path)

    def __len__(self):
        return len(self._files_path)

    @property
    def root(self):
        return self._root
