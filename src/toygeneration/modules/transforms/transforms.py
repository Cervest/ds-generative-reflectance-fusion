from abc import ABC, abstractmethod
import numpy as np
import random
from PIL import Image
import imgaug.augmenters as iaa
import imgaug.parameters as iap
from skimage.transform import PiecewiseAffineTransform, warp
from src.utils import setseed


class Transformer(ABC):
    """Base class for custom transformers objects which used within declaration
    of a dataloader object in 'transforms' field

    Args:
        mode (str): transformer mode specification
    """

    def __init__(self, mode=None):
        self.mode = mode

    @abstractmethod
    def __call__(self, img):
        raise NotImplementedError

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


class ToNumpy(Transformer):
    """convert to numpy array
    """
    def __call__(self, img):
        return np.asarray(img)


# class ToPIL(Transformer):
#     """convert as PIL Image
#     """
#     def __init__(self, *args, mode, **kwargs):
#         super(ToPIL, self).__init__(*args, **kwargs)
#         self.mode = mode
#
#     def __call__(self, img):
#         return F.to_pil_image(img, mode=self.mode)


class RandomScale(Transformer):
    """Rescales PIL Image with random scaling factor

    Args:
        scale (tuple[int]): (min_scale, max_scale)
    """
    def __init__(self, scale):
        super().__init__(mode=None)
        self._scale = scale

    def __call__(self, img):
        scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        new_width = int(img.size[0] * scale)
        new_height = int(img.size[1] * scale)
        return img.resize(size=(new_width, new_height))

    @property
    def scale(self):
        return self._scale


class TangentialScaleDistortion(iaa.Augmenter):
    """Emulation of imagery tangential distortion with piecewise affine
    transformation

    Args:
        image_size (tuple[int]): (width, height)
        mesh_size (tuple[int]): (n_cells_columns, n_cells_rows) number of mesh cells in
            rows and columns for piecewise affine morphing
        axis (int): distortion axis {height/rows: 0, width/columns: 1}
        growth_rate (float): sigmoid growth rate parameter
            (default : 4 / length_distortion_axis)
    """
    def __init__(self, image_size, mesh_size, axis=0, growth_rate=None):
        super().__init__(name='tangential_scale_distortion')
        self.axis = axis
        self._swath_length = image_size[axis]
        self._growth_rate = growth_rate or 4 / self.swath_length
        self._transform = self._build_transform(image_size=image_size,
                                                mesh_size=mesh_size,
                                                axis=axis)

    def _build_source_meshgrid(self, image_size, mesh_size):
        """Creates meshgrids of image size and number of mesh specified
        Args:
            image_size (tuple[int]): (height, width)
            mesh_size (tuple[int]): (n_cells_rows, n_cells_columns) number of mesh cells in
                rows and columns for piecewise affine morphing
        Returns:
            type: np.ndarray
        """
        # Build source meshgrid
        h, w = image_size
        src_rows = np.linspace(0, h, mesh_size[0])
        src_cols = np.linspace(0, w, mesh_size[1])
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        return src

    def _build_target_meshgrid(self, src, axis):
        """Applies sigmoid deformation to source meshgrid to compute
        target meshgrid

        Args:
            src (np.ndarray): source meshgrid
            axis (int): distortion axis {height/rows: 0, width/columns: 1}

        Returns:
            type: np.ndarray
        """
        # Apply deformation on specified axis to obtain target meshgrid
        if axis == 0:
            tgt_rows = self._deform_axis(src[:, 1])
            tgt_cols = src[:, 0]

        elif axis == 1:
            tgt_rows = src[:, 1]
            tgt_cols = self._deform_axis(src[:, 0])

        tgt = np.vstack([tgt_cols, tgt_rows]).T
        bounds = np.array([np.min(tgt_rows), np.max(tgt_rows),
                           np.min(tgt_cols), np.max(tgt_cols)], dtype=np.int)
        return tgt, bounds

    def _deform_axis(self, coordinates):
        """Applies edges sigmoid compression of coordinates

        Args:
            coordinates (np.ndarray)
        """
        nadir = 0.5 * self.swath_length
        return self.swath_length * self.sigmoid((coordinates - nadir), self.growth_rate)

    def _build_transform(self, image_size, mesh_size, axis):
        """Builds source and target meshgrids on image with sigmoid
        compression of edges and fits piecewise affine transform on meshgrids

        Inspired from : https://scikit-image.org/docs/dev/auto_examples/transform/
        plot_piecewise_affine.html#sphx-glr-auto-examples-transform-plot-piecewise-affine-py

        Args:
            image_size (tuple[int]): (width, height)
            mesh_size (tuple[int]): (n_cells_columns, n_cells_rows) number of mesh cells in
                rows and columns for piecewise affine morphing
            axis (int): distortion axis {width/rows: 1, height/columns: 0}

        Returns:
            type: PiecewiseAffineTransform
        """
        # Build source meshgrid
        src = self._build_source_meshgrid(image_size=image_size,
                                          mesh_size=mesh_size)

        # Apply deformation on specified axis to obtain target meshgrid
        tgt, bounds = self._build_target_meshgrid(src=src,
                                                  axis=axis)
        self.bounds = bounds

        # Fit piecewise affine transform
        transform = PiecewiseAffineTransform()
        transform.estimate(tgt, src)
        return transform

    def augment_image(self, image):
        """Wraps transform on img and crop at new size
        Args:
            image (np.ndarray)

        Returns:
            type: np.ndarray
        """
        image = warp(image, self.transform)
        image = image[self.bounds[0]: self.bounds[1],
                      self.bounds[2]: self.bounds[3]]
        return image

    def _augment_images(self, images):
        """Extends augment_image to list of images
        Method added for sole purpose of compatibility with iaa.augmenters features
        such as iaa.Sequential
        """
        return [self.augment_image(image=image) for image in images]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        """
        Method added for sole purpose of compatibility with iaa.augmenters features
        such as iaa.Sequential
        """
        if batch.images is not None:
            batch.images = self._augment_images(batch.images)
        return batch

    @staticmethod
    def sigmoid(x, r):
        return 1 / (1 + np.exp(-r * x))

    def get_parameters(self):
        return self.__dict__

    @property
    def axis(self):
        return self._axis

    @property
    def swath_length(self):
        return self._swath_length

    @property
    def growth_rate(self):
        return self._growth_rate

    @property
    def transform(self):
        return self._transform

    @property
    def bounds(self):
        return self._bounds

    @axis.setter
    def axis(self, axis):
        if axis in {0, 1}:
            self._axis = axis
        else:
            raise ValueError("Axis must be in {0, 1}")

    @bounds.setter
    def bounds(self, bounds):
        self._bounds = bounds


class SaltAndPepper(iaa.ReplaceElementwise):
    """
    Straight from https://imgaug.readthedocs.io/en/latest/_modules/imgaug/augmenters/arithmetic.html#SaltAndPepper
    but changed using replacement in [0, 1] to comply with numpy arrays value range
    """
    def __init__(self, p=(0.0, 0.03), per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(SaltAndPepper, self).__init__(
            mask=p,
            replacement=iap.Beta(0.5, 0.5),
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class MultiplyAndAdd(iaa.Augmenter):
    """Draws static random scaling and bias scalars (w, b). Any input array x
        is transformed as w*x + b

    Args:
        mul (tuple[int]): (min_mult_factor, max_mult_factor)
        add (tuple[int]): (min_bias_factor, max_bias_factor)
        seed (int): random seed
    """
    @setseed('numpy')
    def __init__(self, mul=(0.7, 1.3), add=(-0.1, 0.1), seed=None):
        super().__init__(name='multiply_and_add', seed=seed)
        self._w = np.random.rand() * (mul[1] - mul[0]) + mul[0]
        self._b = np.random.rand() * (add[1] - add[0]) + add[0]

    def augment_image(self, image):
        return self.w * image + self.b

    def _augment_images(self, images):
        return [self.augment_image(image=image) for image in images]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is not None:
            batch.images = self._augment_images(batch.images)
        return batch

    def get_parameters(self):
        return self.__dict__

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b


class Patcher:
    """Callable that transplants patch image onto a background reference image

    Args:
        patch_path (str): path to patch image to load
        loc (tuple[int]): patch left-upper corner location in patched image
        scale (float): proportion of image area that should be covered by patch
        mode (str): image channels mode in {'L','RGB','CMYK'}
        return_bbox (bool): if true, returns patch surrounding bounding box
    """
    def __init__(self, patch_path, loc=(0, 0), scale=1, mode='RGBA', return_bbox=False):
        if patch_path:
            self._patch = Image.open(patch_path).convert(mode)
        else:
            self._path = patch_path
        self._loc = loc
        self._scale = scale
        self._mode = mode
        self._return_bbox = return_bbox

    def __call__(self, img, patch=None, loc=None, scale=None):
        """Transplants self.patch over image
        Args:
            img (PIL.Image.Image): pillow image object to patch
            patch (PIL.Image.Image): image patch
            loc (tuple[int]): patch left-upper corner location in patched image
            scale (float): proportion of image area that should be covered by patch
        Returns:
            type: PIL.Image.Image
        """
        # Setup vars
        patch = patch or self.patch
        loc = loc or self.loc
        scale = scale or self.scale

        # Create copy of image
        patched_img = img.copy()

        # Rescale preserving aspect ratio
        patch = self.rescale_to_image(img, patch, scale)

        # Paste rescaled patch at specified location on input image
        patched_img.paste(patch, loc, mask=patch)

        if self.return_bbox:
            x, y = loc
            pw, ph = patch.size
            return patched_img, (x, y, x + pw, x + ph)
        else:
            return patched_img

    def rescale_to_image(self, img, patch, scale=None):
        """Rescales patch as to cover the proportion of the image specified
        by scale parameter

        Args:
            img (PIL.Image.Image): reference image
            patch (PIL.Image.Image)
            scale (float): image scale to be occupied

        Returns:
            type: PIL.Image.Image
        """
        # Setup vars and copy images
        width_img, height_img = img.size
        scale = scale or self.scale
        patch = patch.copy()
        pw, ph = patch.size

        # Rescale preserving aspect ratio
        if pw > ph:
            new_pw = width_img * scale
            new_ph = ph * new_pw / pw
        else:
            new_ph = height_img * scale
            new_pw = pw * new_ph / ph
        new_pw, new_ph = int(new_pw), int(new_ph)

        patch = patch.resize((new_pw, new_ph))
        return patch

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def patch(self):
        return self._patch

    @property
    def mode(self):
        return self._mode

    @property
    def return_bbox(self):
        return self._return_bbox

    def set_patch(self, patch_path):
        self._patch = Image.open(patch_path).convert(self.mode)

    def set_loc(self, loc):
        self._loc = loc

    def set_scale(self, scale):
        self._scale = scale

    def set_mode(self, mode):
        self._mode = mode

    def set_return_bbox(self, return_bbox):
        self._return_bbox = return_bbox


class RandomPatcher(Patcher):
    """Short summary.

    Args:
        patch_path (str): path to patch image to load
        loc (tuple[tuple[int]]): left-upper and lower-right corners of allowed
            patching area, for example ((10, 10), (68, 68))
        scale (tuple[float]): min and max proportion of image area that should be covered by patch
        loc_pdf (callable): Description of parameter `loc_pdf`.
        scale_pdf (callable): Description of parameter `scale_pdf`.
        mode (str): image channels mode in {'L','RGB','CMYK'}
        return_bbox (bool): if true, returns patch surrounding bounding box
    """

    def __init__(self, patch_path, loc, scale, loc_pdf=np.random.rand,
                 scale_pdf=np.random.rand, mode='RGBA', return_bbox=False):
        self.super().__init__(self, patch_path, loc, scale, mode, return_bbox)
        self._loc_pdf = loc_pdf
        self._scale_pdf = scale_pdf

    @setseed('numpy')
    def _rdm_loc(self, seed):
        up_bound, left_bound = self.loc[0]
        low_bound, right_bound = self.loc[1]

        # Compute random location according to specified pdf
        x = int((right_bound - left_bound) * self.loc_pdf()) + left_bound
        y = int((low_bound - up_bound) * np.loc_pdf()) + up_bound

        return x, y

    @setseed('numpy')
    def _rdm_scale(self, seed):
        low_bound, up_bound = self.scale
        scale = (up_bound - low_bound) * self.scale_pdf() + low_bound
        return scale

    def __call__(self, img, patch=None, seed=None):
        loc = self._rdm_loc(seed)
        scale = self._rdm_scale(seed)
        return super().__call__(self, img, patch, loc, scale)

    @property
    def loc_pdf(self):
        return self._loc_pdf

    @property
    def scale_pdf(self):
        return self._scale_pdf
