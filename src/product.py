from PIL import Image
import numpy as np
import random
from progress.bar import Bar
from .export import ProductExport
from src.utils import setseed


class Product(dict):
    """Plain product class, composed of a background image and multiple binary blobs

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
        annotation_bands (int): number of bands of the product annotation mask
        mode (str): patching strategy in {'random', 'grid'}
        grid_size (tuple[int]): grid cells dimensions as (width, height)
        color (int, tuple[int]): color value for background (0-255) according to mode
        blob_transform (callable): geometric transformation to apply blobs when patching
        rdm_dist (callable): numpy random distribution to use for randomization
        seed (int): random seed
        blobs (dict): hand made dict formatted as {idx: (location, blob)}
    """
    __mode__ = {'random', 'grid'}

    def __init__(self, size, horizon=None, nbands=1, annotation_bands=2,
                 mode='random', grid_size=None, color=0, blob_transform=None,
                 rdm_dist=np.random.rand, seed=None, blobs={}):
        super(Product, self).__init__(blobs)
        self._size = size
        self._nbands = nbands
        self._annotation_bands = annotation_bands
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

    @setseed('numpy', 'random')
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
            - Same number of annotation mask channels
        Args:
            blob (BinaryBlob)
        """
        # Verify matching number of bands
        assert blob.ndim == self.nbands, f"""Trying to add {blob.ndim}-dim blob
            while product is {self.nbands}-dim"""

        # Verify time serie horizon at least equals produc horizon
        if blob.time_serie is not None:
            assert blob.time_serie.horizon >= self.horizon, \
                f"""blob has {blob.time_serie.horizon} horizon while product has
                 a {self.horizon} horizon"""

        # Verify nb of annotation bands checks out with blob attributes
        if self.annotation_bands == 2:
            assert blob.time_serie is not None, """Trying to add a blob
                without time serie while product expects one"""
        elif self.annotation_bands == 1:
            assert blob.time_serie is None, """Trying to add a blob
                with time serie while product does not expects one"""
        else:
            raise ValueError("Number of annotation channels should be in {1, 2}")
        return True

    def register(self, blob, loc, seed=None):
        """Registers blob

        Args
            blob (BinaryBlob): blob instance to register
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
            idx = len(self) + 1
            blob.set_idx(idx)

        # Apply product defined random geometric augmentation
        blob = self._augment_blob(blob=blob, seed=seed)
        self[idx] = (loc, blob)
        blob.affiliate()

    @setseed('numpy')
    def random_register(self, blob, seed=None):
        """Registers blob to product applying random strategy
        for the choice of its patching location

        Args:
            blob (BinaryBlob): blob instance to register
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
                raise IndexError("Trying to register too many blobs, no space left on grid")
        self.register(blob, loc, seed)

    @setseed('random')
    def _augment_blob(self, blob, seed=None):
        """If defined, applies transformation to blob
        Args:
            blob (BinaryBlob)
            seed (int): random seed (default: None)
        Returns:
            type: blob
        """
        # If product defines blobs transformation, use it
        if self.blob_transform:
            augmented_blob = self.blob_transform(blob)
            augmented_blob = blob._new(augmented_blob.im)
        # Else use blob as is
        else:
            augmented_blob = blob
        return augmented_blob

    def view(self):
        """Generates grayscale image of background with patched blobs
        Returns:
            type: PIL.Image.Image
        """
        # Copy background image
        img = self.bg.copy()
        for loc, blob in self.values():
            # Compute upper-left corner position
            upperleft_loc = self.center2upperleft(loc, blob.size)
            # Paste on background with transparency mask
            img.paste(blob, upperleft_loc, mask=blob)
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

    def generate(self, output_dir, astype='h5'):
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
            astype (str): in {'h5', 'jpg'}
        """
        # Prepare product and export
        self.prepare()
        export = ProductExport(output_dir, astype)
        export._setup_output_dir()
        export._init_generation_index(self)
        bar = Bar("Generation", max=self.horizon)

        for i in range(self.horizon):
            # Create copies of background to preserve original
            img = self.bg.array.copy()
            annotation = np.zeros(self.bg.size + (self.annotation_bands,))

            for idx, (loc, blob) in self.items():
                # Update blob in size and pixel values
                patch, annotation_mask = next(blob)
                # Patch on background
                self.patch_array(img, patch, loc)
                self.patch_array(annotation, annotation_mask, loc)

            frame_name = '.'.join([f"frame_{i}", astype])
            annotation_name = f"annotation_{i}.h5"

            # Record in index
            export.add_to_index(i, frame_name, annotation_name)

            # Dump file
            export.dump_frame(img, frame_name)
            export.dump_annotation(annotation.astype(np.int16), annotation_name)
            bar.next()

        # Save index
        export.dump_index()

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
    def _crop_patch(bg_array, patch_array, loc):
        """Crops patch given background and patching location

        Args:
            bg_array (np.ndarray): background array, valued in [0, 1]
            patch_array (np.ndarray): array to patch, valued in [0, 1]
            loc (tuple[int]): patching location

        Returns:
            type: np.ndarray, int, int, int, int
        """
        height, width = patch_array.shape[:2]
        upperleft_loc = Product.center2upperleft(loc, (width, height))
        y, x = upperleft_loc
        # Crop patch if out-of-bounds upper-left patching location
        if x < 0:
            patch_array = patch_array[-x:]
            x = 0
        if y < 0:
            patch_array = patch_array[:, -y:]
            y = 0
        w, h = patch_array.shape[:2]

        # Again crop if out-of-bounds lower-right patching location
        w = min(w, bg_array.shape[0] - x)
        h = min(h, bg_array.shape[1] - y)
        return patch_array, x, y, w, h

    @staticmethod
    def patch_array(bg_array, patch_array, loc):
        """Patching of numpy array into another numpy array
        Patch is cropped if needed to handle out-of-boundaries patching

        Args:
            bg_array (np.ndarray): background array, valued in [0, 1]
            patch_array (np.ndarray): array to patch, valued in [0, 1]
            loc (tuple[int]): patching location

        Returns:
            type: np.ndarray
        """
        # Crop it
        patch_array, x, y, w, h = Product._crop_patch(bg_array, patch_array, loc)

        # Patch it
        mask = patch_array[:w, :h] > 0
        bg_array[x:x + w, y:y + h][mask] = patch_array[:w, :h][mask].flatten()
        return bg_array

    @staticmethod
    def center2upperleft(loc, patch_size):
        y, x = loc
        w, h = patch_size
        upperleft_loc = (y - w // 2, x - h // 2)
        return upperleft_loc

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
    def annotation_bands(self):
        return self._annotation_bands

    @property
    def horizon(self):
        return self._horizon

    @property
    def seed(self):
        return self._seed
