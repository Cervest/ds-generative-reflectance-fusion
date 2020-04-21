from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
from PIL import Image


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
    """torchvision.utils.transforms like class to convert PIL format to numpy
    """
    def __call__(self, img):
        return np.asarray(img)


class Patcher(Transformer):
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

    @setseed
    def _rdm_loc(self, seed):
        up_bound, left_bound = self.loc[0]
        low_bound, right_bound = self.loc[1]

        # Compute random location according to specified pdf
        x = int((right_bound - left_bound) * self.loc_pdf()) + left_bound
        y = int((low_bound - up_bound) * np.loc_pdf()) + up_bound

        return x, y

    @setseed
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
