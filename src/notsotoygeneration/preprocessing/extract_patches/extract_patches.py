"""
Description

Usage: extract_patches.py --shapefile=<path_to_shapefile> --o=<output_directory> --modis_root=<modis_scenes_directory> [--modis_list=<modis_scenes_list>] --s2_root=<s2_scenes_directory> [--s2_list=<s2_scenes_list>] --s1_root=<s1_scenes_directory> [--s1_list=<s1_scenes_list>]

Options:
  -h --help                                 Show help.
  --version                                 Show version.
  --shapefile=<raw_files_directory>         Path to shape file
  --o=<output_directory>                    Output directory
  --scenes_root=<path_to_scenes_directory>  Path to directory containing scenes to chip
  --scenes_list=<path_to_scenes_list>       Path to file with list of scenes to chip
"""
import os
import sys
from docopt import docopt
import logging
import numpy as np
from shapely import geometry
from PIL import Image
from progress.bar import Bar
import rasterio

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../")
sys.path.append(base_dir)

from src.notsotoygeneration.io import readers


def main(args):
    # Instantiate readers
    s2_reader = readers.S2SceneReader(root=args['--s2_root'])
    s1_reader = readers.S1SceneReader(root=args['--s1_root'])
    modis_reader = readers.MODISSceneReader(root=args['--modis_root'])

    # Match together tiles coordinates that have an intersecting area
    matching_locations = match_and_align(s2_scenes_list, s1_scenes_list, modis_scenes_list)

    for coordinates, bbox in matching_locations:
        s2_coordinate, s1_coordinate, modis_coordinate = coordinates

        for date in s2_scenes_list[s2_coordinate]['dates']:
            s2_raster = s2_reader.open(coordinate=s2_coordinate, date=date)
            s2_raster = crop_to_bbox_and_resample(raster=s2_raster, bbox=bbox, resolution=target_resolution)
            for idx, patch_array in cut_patch_from_raster(s2_raster, windows):
                export.save(patch=patch_array, root=s2_dump_root, coordinate=s2_coordinate, date=s2_date)

        for date in s1_scenes_list[s1_coordinate]['dates']:
            s1_raster = s1_reader.open(coordinate=s1_coordinate, date=s1_date)
            s1_raster = crop_to_bbox_and_resample(raster=s1_raster, bbox=bbox, resolution=target_resolution)
            for idx, patch_array in cut_patch_from_raster(s1_raster, windows):
                export.save(patch=patch_array, root=s1_dump_root, coordinate=s1_coordinate, date=s1_date)

        for date in modis_scenes_list[modis_coordinate]['dates']:
            modis_raster = modis_reader.open(coordinate=modis_coordinate, date=modis_date)
            modis_raster = crop_to_bbox_and_resample(raster=modis_raster, bbox=bbox, resolution=target_resolution)
            for idx, patch_array in cut_patch_from_raster(modis_raster, windows):
                export.save(patch=patch_array, root=modis_dump_root, coordinate=modis_coordinate, date=modis_date)


def extract_and_dump_patches(reader, scenes_specs, coordinate, bbox, resolution, patch_size, dump_root):
    """Handles patch extraction from raw reprojected raster to dumped patch arrays
        1 - Loads rasters corresponding to specified scenes
        2 - Crops following bounding box dimension
        3 - Resamples to target resolution
        4 - Extracts and dumps patches as .h5 files

    Args:
        reader (SceneReader): scene reading utility
        scenes_specs (dict): scenes to load specification dictionnary
        coordinate (object): coordinate information - depends on scene reader
        bbox (shapely.geometry.Box): cropping bounding box for raster
        resolution (tuple[float]): (height_resolution, width_resolution)
        patch_size (tuple[int]): (patch_height, patch_width) in pixels
        dump_root (str): root dumping directory for patches
    """
    for date in scenes_specs[coordinate]['dates']:
        s2_raster = reader.open(coordinate=coordinate, date=date)
        s2_raster = crop_to_bbox_and_resample(raster=raster, bbox=bbox, resolution=resolution)
        for patch_array in cut_patch_from_raster(raster=raster, patch_size=patch_size):
            export.save(patch=patch_array, root=dump_root, coordinate=coordinate, date=date)
