"""
Runs joint registered patch extraction on MODIS and Landsat rasters of same location for several dates

Usage: extract_patches_modis_landsat.py --o=<output_directory> --modis_root=<modis_scenes_directory>  --landsat_root=<landsat_scenes_directory> --scenes_specs=<scenes_to_load>

Options:
  -h --help                                  Show help.
  --version                                  Show version.
  --shapefile=<raw_files_directory>          Path to shape file
  --o=<output_directory>                     Output directory
  --modis_root=<path_to_scenes_directory>    Path to directory containing scenes to chip
  --landsat_root=<path_to_scenes_directory>  Path to directory containing scenes to chip
  --scenes_specs=<scenes_to_load>            Path to list of scenes to load
"""
import os
import sys
import yaml
from docopt import docopt
import logging
from itertools import product
from functools import reduce
from operator import add
import numpy as np
from shapely import geometry, affinity
from rasterio.windows import Window
from progress.bar import Bar

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../")
sys.path.append(base_dir)

from src.notsotoygeneration.io import readers
from src.notsotoygeneration.preprocessing import utils
from src.notsotoygeneration.preprocessing.patch_extraction import PatchExport


def main(args):
    # Instantiate readers and exporter
    landsat_reader = readers.LandsatSceneReader(root=args['--landsat_root'])
    modis_reader = readers.MODISSceneReader(root=args['--modis_root'])
    export = PatchExport(output_dir=args['--o'])
    logging.info("Loaded scene readers")

    # Load scenes specifications
    with open(args['--scenes_specs'], 'r') as f:
        scenes_specs = yaml.safe_load(f)

    # Compute scenes alignement features out of landsat rasters
    intersecting_bbox, max_resolution = compute_registration_features(scenes_specs=scenes_specs,
                                                                      reader=landsat_reader)
    logging.info("Computed registration features")

    for date in scenes_specs['dates']:
        # Load corresponding rasters
        landsat_raster, modis_raster, qa_raster = load_rasters(date=date,
                                                               landsat_reader=landsat_reader,
                                                               modis_reader=modis_reader)

        # Register rasters together
        logging.info(f"Date {date} : Aligning rasters")
        landsat_raster = align_raster(landsat_raster, intersecting_bbox, max_resolution)
        qa_raster = align_raster(qa_raster, intersecting_bbox, max_resolution)
        modis_raster = align_modis_raster(modis_raster, intersecting_bbox, max_resolution)

        # Compute valid pixel map out of landsat quality assessment raster
        logging.info(f"Date {date} : Computing valid pixel map")
        valid_pixels = compute_landsat_raster_valid_pixels_map(qa_raster)

        # Instantiate iterator over raster windows
        windows_iterator = make_windows_iterator(image_size=(landsat_raster.height, landsat_raster.width),
                                                 window_size=scenes_specs['patch_size'],
                                                 valid_pixels=valid_pixels,
                                                 validity_threshold=scenes_specs['validity_threshold'])

        # Run patches extraction and dumping
        bar = Bar(f"Date {date} :  Extracting patches from rasters")
        for patch_idx, window in windows_iterator:
            extract_and_dump_patch(landsat_raster=landsat_raster,
                                   modis_raster=modis_raster,
                                   window=window,
                                   patch_idx=patch_idx,
                                   date=date,
                                   export=export)
            bar.next()


def extract_and_dump_patch(landsat_raster, modis_raster, window, patch_idx, date, export):
    """Handles joined patch extraction and dumping given patch window and export protocol

    Args:
        landsat_reader (rasterio.io.DatasetReader)
        modis_reader (rasterio.io.DatasetReader)
        window (rasterio.windows.Window)
        patch_idx (int)
        date (str): date formatted as yyyy-mm-dd
        export (PatchExport)
    """
    # Set export directories and index
    export.setup_output_dir(patch_idx=patch_idx)
    patch_bounds = list(map(int, reduce(add, map(list, window.toranges()))))
    index = export.setup_index(patch_idx=patch_idx, patch_bounds=patch_bounds)
    export.dump_index(index=index, patch_idx=patch_idx)

    # Extract patches
    modis_patch = modis_raster.read(window=window)
    landsat_patch = landsat_raster.read(window=window)

    # Update export index
    index = export.update_index(index=index,
                                patch_idx=patch_idx,
                                date=date)

    # Dump frames and index
    export.dump_patches(patch_idx=patch_idx,
                        date=date,
                        modis_patch=modis_patch,
                        landsat_patch=landsat_patch)
    export.dump_index(index=index, patch_idx=patch_idx)


def load_rasters(date, landsat_reader, modis_reader):
    """Handles Landsat and MODIS rasters loading along with Landsat QA raster

    Args:
        date (str): date formatted as yyyy-mm-dd
        landsat_reader (LandsatSceneReader)
        modis_reader (MODISSceneReader)

    Returns:
        type: tuple[rasterio.io.DatasetReader]
    """
    landsat_raster = landsat_reader.open(coordinate=198026, date=date)
    qa_raster = landsat_reader.open(coordinate=198026, date=date, is_quality_map=True)
    modis_raster = modis_reader.open(coordinate=(18, 4), date=date)
    return landsat_raster, modis_raster, qa_raster


def compute_registration_features(scenes_specs, reader):
    """In order to extract fully registered patches, rasters must be aligned
    by resampling them to the same resolution and cropping them to the same
    window

    We here compute greatest resolution and tightest bounds among all landsat
    scenes which will define our registering features

    Args:
        scenes_specs (dict): specification of files to load
        reader (SceneReader): scene band reading utility

    Returns:
        type: shapely.geometry.Box, tuple[float]
    """
    # Gather bounds and resolution data for each raster
    bounds = []
    resolutions = []
    for date in scenes_specs['dates']:
        with reader(coordinate=198026, date=date) as raster:
            bounds += [raster.bounds]
            resolutions += [raster.res]

    # Compute intersecting bounding box
    bounds = np.array(bounds)
    max_bounds, min_bounds = np.max(bounds, axis=0), np.min(bounds, axis=0)
    intersecting_bounds = max_bounds[:2].tolist() + min_bounds[2:].tolist()
    intersecting_bbox = geometry.box(*intersecting_bounds)

    # Compute greatest resolution
    max_resolution = np.max(resolutions, axis=0).tolist()
    max_resolution = tuple(max_resolution)
    return intersecting_bbox, max_resolution


def compute_landsat_raster_valid_pixels_map(qa_raster):
    """Given landsat quality assessment raster, computes boolean array of valid
    pixels

    See for QA pixels values :
    https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1368_L8_SurfaceReflectanceCode-LASRC_ProductGuide-v2.pdf

    Args:
        qa_raster (rasterio.io.DatasetReader): Landsat QA raster

    Returns:
        type: np.ndarray

    """
    # Define list of valid quality assessment pixel values
    clear_values = [322, 386, 834, 898, 1346]
    water_values = [324, 388, 836, 900, 1328]
    cloud_shadow = [328, 392, 840, 904, 1350]
    snow_ice = [336, 368, 400, 432, 848, 880, 912, 944, 1352]
    valid_values = clear_values + water_values + cloud_shadow + snow_ice

    # Generate boolean mask of pixels belonging to allowed values
    qa_array = qa_raster.read(1)
    valid_pixels = np.in1d(qa_array.flatten(), valid_values)
    valid_pixels = valid_pixels.reshape(qa_raster.height, qa_raster.width)
    return valid_pixels


def align_raster(raster, cropping_bbox, target_resolution):
    """Resamples and crops raster to specified resolution and bounding box

    Args:
        raster (rasterio.io.DatasetReader)
        cropping_bbox (shapely.geometry.Box)
        target_resolution (tuple[float])

    Returns:
        type: rasterio.io.DatasetReader
    """
    # Resample raster to target resolution
    resampled_raster = utils.resample_raster(raster, target_resolution, as_raster=True)

    # Finally crop resampled raster to target bounding box
    cropped_raster = utils.crop_raster_to_bbox(resampled_raster, cropping_bbox, as_raster=True)
    return cropped_raster


def align_modis_raster(modis_raster, cropping_bbox, target_resolution):
    """MODIS rasters require an additional step ahead of resampling given its low resolution :

        - If resampled as is, output raster becomes way too large
        - Cropping to target bbox before resampling hampers resampling quality
            because there are to few pixels

    We hence propose to pre-crop modis raster to a window twice as large as the
    target bounding box and then only proceed with resampling + cropping


    Args:
        modis_raster (rasterio.io.DatasetReader)
        cropping_bbox (shapely.geometry.Box)
        target_resolution (tuple[float])

    Returns:
        type: rasterio.io.DatasetReader
    """
    # Rescale box to area larger than bounding box for faster and still accurate resampling
    resampling_bbox = affinity.scale(cropping_bbox, xfact=2, yfact=2)
    cropped_raster = utils.crop_raster_to_bbox(modis_raster, resampling_bbox, as_raster=True)

    # Align modis raster to same resolution bounding box
    aligned_raster = align_raster(cropped_raster, cropping_bbox, target_resolution)
    return aligned_raster


def make_windows_iterator(image_size, window_size, valid_pixels, validity_threshold):
    """Iterates of patch windows corresponding to valid pixel positions

    Args:
        image_size (tuple[int]): (height, width) in pixels
        window_size (tuple[int]): (window_height, window_width) in pixels
        valid_pixels (np.ndarray): (height, width) boolean array
        validity_threshold (float): percentage of valid pixels in a window to be considered valid

    Yields:
        type: rasterio.windows.Window

    """
    # Make full raster window object
    height, width = image_size
    full_image_window = Window(col_off=0, row_off=0,
                               width=width, height=height)

    # Create offsets range to iterate on
    window_height, window_width = window_size
    col_row_offsets = product(range(0, width, window_width),
                              range(0, height, window_height))

    for window_idx, (col_offset, row_offset) in enumerate(col_row_offsets):
        # Create window instance
        window = Window(col_off=col_offset, row_off=row_offset,
                        width=window_width, height=window_height)

        # Verify the window is valid
        window_valid_pixels = valid_pixels[window.toslices()]
        is_valid = window_valid_pixels.sum() / window_valid_pixels.size > validity_threshold
        if not is_valid:
            continue

        # Intersect and return window
        window = window.intersection(full_image_window)
        yield window_idx, window


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'arguments: {args}')

    # Run generation
    main(args)
