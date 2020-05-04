from PIL import Image
import numpy as np
import random
from progress.bar import Bar
from .export import ProductExport
from src.utils import setseed


class Product(dict):
    """Plain product class, composed of a background image and multiple digits

    > Registers digits as a dictionnary {idx: (location_on_bg, digit)}
    > Proposes fully random and grid based patching strategy for digits
    > Generates view of patched image on the fly

    - 'random' mode : each digit location is computed as the product height and
        width scaled by the specified random distribution. Better used with
        distribution valued in [0, 1]
    - 'grid' mode : a regular grid perturbed by the specifed random distribution
        is used to determine digits locations. Better used with centered distributions

    Args:
        size (tuple[int]): (width, height) for background
        horizon (int): number of time steps to generate
        nbands (int): number of bands of the product
        annotation_bands (int): number of bands of the product annotation mask
        mode (str): patching strategy in {'random', 'grid'}
        grid_size (tuple[int]): grid cells dimensions as (width, height)
        color (int, tuple[int]): color value for background (0-255) according to mode
        digit_transform (callable): geometric transformation to apply digits when patching
        rdm_dist (callable): numpy random distribution to use for randomization
        seed (int): random seed
        digits (dict): hand made dict formatted as {idx: (location, digit)}
    """
    __mode__ = {'random', 'grid'}

    def __init__(self, size, horizon=None, nbands=1, annotation_bands=2,
                 mode='random', grid_size=None, color=0, digit_transform=None,
                 rdm_dist=np.random.rand, seed=None, digits={}):
        super(Product, self).__init__(digits)
        self._size = size
        self._nbands = nbands
        self._annotation_bands = annotation_bands
        self._horizon = horizon
        self._mode = mode
        self._bg = Image.new(size=size, color=color, mode='L')
        self._digit_transform = digit_transform
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
            favor scattered patching location when nb of digits < nb locations

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

    def _assert_compatible(self, digit):
        """Ensure digit is compatible with product verifying it has :
            - Same number of bands / dimensionality
            - Same or greater horizon
            - Same number of annotation mask channels
        Args:
            digit (Digit)
        """
        # Verify matching number of bands
        assert digit.ndim == self.nbands, f"""Trying to add {digit.ndim}-dim digit
            while product is {self.nbands}-dim"""

        # Verify time serie horizon at least equals produc horizon
        if digit.time_serie is not None:
            assert digit.time_serie.horizon >= self.horizon, \
                f"""Digit has {digit.time_serie.horizon} horizon while product has
                 a {self.horizon} horizon"""

        # Verify nb of annotation bands checks out with digit attributes
        if self.annotation_bands == 2:
            assert digit.time_serie is not None, """Trying to add a digit
                without time serie while product expects one"""
        elif self.annotation_bands == 1:
            assert digit.time_serie is None, """Trying to add a digit
                with time serie while product does not expects one"""
        else:
            raise ValueError("Number of annotation channels should be in {1, 2}")
        return True

    def register(self, digit, loc, seed=None):
        """Registers digit

        Args:
            digit (Digit): digit instance to register
            loc (tuple[int]): upper-left corner if 2-tuple, upper-left and
                lower-right corners if 4-tuple
        """
        # Ensure digit dimensionality and horizon match product's
        self._assert_compatible(digit)

        # If digit has an idx, use it
        if digit.idx:
            idx = digit.idx
        # Else create a new one
        else:
            idx = len(self)
            digit.set_idx(idx)

        # Apply product defined random geometric augmentation
        digit = self._augment_digit(digit=digit, seed=seed)
        self[idx] = (loc, digit)
        digit.affiliate()

    @setseed('numpy')
    def random_register(self, digit, seed=None):
        """Registers digit to product applying random strategy
        for the choice of its patching location

        Args:
            digit (Digit): digit instance to register
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
                raise IndexError("Trying to register too many digits, no space left on grid")
        self.register(digit, loc, seed)

    @setseed('random')
    def _augment_digit(self, digit, seed=None):
        """If defined, applies transformation to digit
        Args:
            digit (Digit)
            seed (int): random seed (default: None)
        Returns:
            type: digit
        """
        # If product defines digits transformation, use it
        if self.digit_transform:
            augmented_digit = self.digit_transform(digit)
            augmented_digit = digit._new(augmented_digit.im)
        # Else use digit as is
        else:
            augmented_digit = digit
        return augmented_digit

    def view(self):
        """Generates grayscale image of background with patched digits
        Returns:
            type: PIL.Image.Image
        """
        # Copy background image
        img = self.bg.copy()
        for loc, digit in self.values():
            # Compute upper-left corner position
            upperleft_loc = self.center2upperleft(loc, digit.size)
            # Paste on background with transparency mask
            img.paste(digit, upperleft_loc, mask=digit)
        return img

    def prepare(self):
        """Prepares product for generation by unfreezing digits and setting up
        some hidden cache attributes
        iteration
        """
        for _, digit in self.values():
            digit.unfreeze()
        # Save array version of background in cache
        bg_array = np.expand_dims(self.bg, -1)
        bg_array = np.tile(bg_array, self.nbands).astype(np.float64)
        self.bg.array = bg_array

    def generate(self, output_dir, astype='h5'):
        """Runs generation as two for loops :
        ```
        for time_step in horizon:
            for digit in registered_digits:
                Scale digit with its next time serie slice
                Patch digit on background
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

            for idx, (loc, digit) in self.items():
                # Update digit in size and pixel values
                patch, annotation_mask = next(digit)
                # Patch on background
                self.patch_array(img, patch, loc)
                self.patch_array(annotation, annotation_mask, loc)

            frame_name = '.'.join([f"frame_{i}", astype])
            annotation_name = f"annotation_{i}.h5"

            # Record in index
            export.add_to_index(i, frame_name, annotation_name)

            # Dump file
            export.dump_frame(img, frame_name)
            export.dump_annotation(annotation, annotation_name)
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
        upperleft_loc = Product.center2upperleft(loc, patch_array.shape[:2])
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

        # Patch and clip
        bg_array[x:x + w, y:y + h] += patch_array[:w, :h]
        bg_array.clip(max=1)

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
    def digit_transform(self):
        return self._digit_transform

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
