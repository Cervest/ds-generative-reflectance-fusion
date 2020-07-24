from typing import Tuple, Generator, List, Union

import rasterio.windows
from rasterio.windows import Window
from shapely.geometry import Polygon
import affine
import warnings
from pathlib import Path

import itertools
import numpy as np
import rasterio
import shapely
from PIL import Image as pilimg
from tqdm import tqdm


def get_chip_windows(raster_width: int,
                     raster_height: int,
                     raster_transform,
                     chip_width: int=256,
                     chip_height: int=256,
                     skip_partial_chips: bool=False,
                     ) -> Generator[Tuple[Window, affine.Affine, Polygon], any, None]:
    """
    Generator for rasterio windows of specified pixel size to iterate over an image in chips.
    Chips are created row wise, from top to bottom of the raster.
    :param raster_width: rasterio meta['width']
    :param raster_height: rasterio meta['height']
    :param raster_transform: rasterio meta['transform']
    :param chip_width: Desired pixel width.
    :param chip_height: Desired pixel height.
    :param skip_partial_chips: Skip image chips at the edge of the raster that do not result in a full size chip.
    :return: Yields tuple of rasterio chip window, chip transform and chip polygon.
    """
    col_row_offsets = itertools.product(range(0, raster_width, chip_width), range(0, raster_height, chip_height))
    raster_window = Window(col_off=0, row_off=0, width=raster_width, height=raster_height)

    for col_off, row_off in col_row_offsets:
        chip_window = Window(col_off=col_off, row_off=row_off, width=chip_width, height=chip_height)

        if skip_partial_chips:
            if row_off + chip_height > raster_height or col_off + chip_width > raster_width:
                continue

        chip_window = chip_window.intersection(raster_window)
        chip_transform = rasterio.windows.transform(chip_window, raster_transform)
        chip_bounds = rasterio.windows.bounds(chip_window, raster_transform)  # Uses transform of full raster.
        chip_poly = shapely.geometry.box(*chip_bounds, ccw=False)

        yield (chip_window, chip_transform, chip_poly)


def iter_chip_images(raster, patch_windows):
    """
    Cuts image raster to patches via the given windows and exports them to jpg.
    """

    src = raster

    for chip_window in tqdm(patch_windows):
        img_array = np.dstack(list(src.read(window=chip_window)))
        img_array = np.nan_to_num(img_array)
        img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype('uint8')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # img_array = img_as_ubyte(img_array)
        yield img_array
    src.close()
