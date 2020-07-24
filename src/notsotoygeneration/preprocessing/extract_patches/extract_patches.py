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
import json
import geopandas as gpd
from PIL import Image
from progress.bar import Bar
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.enums import Resampling

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../")
sys.path.append(base_dir)

from src.notsotoygeneration.io import readers
from src.notsotoygeneration.preprocessing.extract_patches.patching import chip_shapes_into_patches
from src.notsotoygeneration.preprocessing.extract_patches.utils.img import iter_chip_images
from src.notsotoygeneration.preprocessing.utils import daterange, get_closest_date


def main(args):
    # Instantiate readers
    s2_reader = readers.S2SceneReader(root=args['--s2_root'])
    s1_reader = readers.S1SceneReader(root=args['--s1_root'])
    modis_reader = readers.MODISSceneReader(root=args['--modis_root'])

    # Load list of scenes to process
    s2_31UEQ_dates, s2_31UFQ_dates, s1_dates, modis_dates = load_scenes_dates()

    # Load shapefile
    gdf = gpd.read_file(args['--shapefile'])

    # Visited dates record
    visited_dates = {'modis': [], 's2_31UEQ': [], 's2_31UFQ': [], 's1': []}

    bar = Bar(f"Time step", max=31)
    for idx, date in enumerate(daterange('2018-05-06', '2018-05-08')):
        modis_date = get_closest_date(modis_dates, date)
        s2_31UEQ_date = get_closest_date(s2_31UEQ_dates, date)
        # s2_31UFQ_date = get_closest_date(s2_31UFQ_dates, date)
        s1_date = get_closest_date(s1_dates, date)

        if idx == 0:
            modis_raster = modis_reader.open(modis_coordinate=(18, 4), date=modis_date)
            s2_31UEQ_raster = s2_reader.open(mgrs_coordinate='31UEQ', date=s2_31UEQ_date)
            # s2_31UFQ_raster = s2_reader.open(mgrs_coordinate='31UFQ', date=s2_31UFQ_date)
            s1_raster = s1_reader.open(date=s1_date)

            bbox_31UEQ = get_intersecting_bbox([modis_raster, s2_31UEQ_raster, s1_raster])
            # bbox_31UFQ = get_intersecting_bbox([modis_raster, s2_31UFQ_raster, s1_raster])

            s2_31UEQ_raster = crop_to_bbox(s2_31UEQ_raster, bbox_31UEQ)
            # s2_31UFQ_raster = crop_to_bbox(s2_31UFQ_raster, bbox_31UFQ)

            patch_dfs_31UEQ, patch_windows_31UEQ = chip_shapes_into_patches(sh_df=gdf, raster=s2_31UEQ_raster,
                                                                            patch_size=256, first_n_patches=1000)
            # patch_dfs_31UFQ, patch_windows_31UFQ = chip_shapes_into_patches(sh_df=gdf, raster=s2_31UFQ_raster,
            #                                                                 patch_size=256, first_n_patches=1000)

        if modis_date not in visited_dates['modis']:
            print("Processing MODIS")
            raster = modis_reader.open(modis_coordinate=(18, 4), date=modis_date)
            raster_31UEQ = crop_and_resample(raster, bbox_31UEQ, s2_31UEQ_raster.res)
            # raster_31UFQ = crop_and_resample(raster, bbox_31UFQ, s2_31UFQ_raster.res)
            chip_raster_into_patches(raster=raster_31UEQ, patch_windows=patch_windows_31UEQ.values(),
                                     dump_root=args['--o'] + '/modis', name_root=f'modis_31UEQ_{idx}')
            # chip_raster_into_patches(raster=raster_31UFQ, patch_windows=patch_windows_31UFQ.values(),
            #                          dump_root=args['--o'] + '/modis', name_root=f'modis_31UFQ_{idx}')
            visited_dates['modis'] += [modis_date]

        # if s1_date not in visited_dates['s1']:
        #     raster = s1_reader.open(date=s1_date)
        #     raster_31UEQ = crop_and_resample(raster, bbox_31UEQ, s2_31UEQ_raster.res)
        #     raster_31UFQ = crop_and_resample(raster, bbox_31UFQ, s2_31UFQ_raster.res)
        #     chip_raster_into_patches(raster=raster_31UEQ, patch_windows=patch_windows_31UEQ.values(),
        #                              dump_root=args['--o'] + '/s1', name_root=f's1_31UFQ_{idx}')
        #     chip_raster_into_patches(raster=raster_31UFQ, patch_windows=patch_windows_31UFQ.values(),
        #                              dump_root=args['--o'] + '/s1', name_root=f's1_31UEQ_{idx}')
        #     visited_dates['s1'] += [s1_date]

        if s2_31UEQ_date not in visited_dates['s2_31UEQ']:
            print("Processing S2")
            raster = s2_reader.open(mgrs_coordinate='31UEQ', date=s2_31UEQ_date)
            raster = crop_to_bbox(raster, bbox_31UEQ)
            chip_raster_into_patches(raster=raster, patch_windows=patch_windows_31UEQ.values(),
                                     dump_root=args['--o'] + '/s2', name_root=f's2_31UEQ_{idx}')
            visited_dates['s2_31UEQ'] += [s2_31UEQ_date]

        # if s2_31UFQ_date not in visited_dates['s2_31UFQ']:
        #     raster = s2_reader.open(mgrs_coordinate='31UFQ', date=s2_31UEQ_date)
        #     raster = crop_to_bbox(raster, bbox_31UFQ)
        #     chip_raster_into_patches(raster=raster, patch_windows=patch_windows_31UFQ.values(),
        #                              dump_root=args['--o'] + '/s2', name_root=f's2_31UFQ_{idx}')
        #     visited_dates['s2_31UFQ'] += [s2_31UFQ_date]

        bar.next()


def load_scenes_dates():
    s2_31UEQ_dates = retrieve_dates_from_aws_structure('/home/shahine/ds-virtual-remote-sensing-toy/data/not-so-toy/reprojected/s2/31/U/EQ')
    s2_31UFQ_dates = retrieve_dates_from_aws_structure('/home/shahine/ds-virtual-remote-sensing-toy/data/not-so-toy/reprojected/s2/31/U/FQ')
    s1_dates = retrieve_dates_from_aws_structure('/home/shahine/ds-virtual-remote-sensing-toy/data/not-so-toy/reprojected/s1')
    modis_dates = retrieve_dates_from_aws_structure('/home/shahine/ds-virtual-remote-sensing-toy/data/not-so-toy/reprojected/modis/18/04/')
    return s2_31UEQ_dates, s2_31UFQ_dates, s1_dates, modis_dates


def crop_and_resample(raster, bbox, resolution):
    raster = crop_to_bbox(raster, bbox)
    raster = resample(raster, resolution)
    return raster


def resample(raster, target_resolution):
    height_resampling = raster.res[0] / target_resolution[0]
    width_resampling = raster.res[1] / target_resolution[1]

    img = raster.read(out_shape=(raster.count,
                                 int(np.ceil(raster.height * height_resampling)),
                                 int(np.ceil(raster.width * width_resampling))),
                      resampling=Resampling.nearest)

    transform = raster.transform * raster.transform.scale((raster.width / img.shape[-1]),
                                                          (raster.height / img.shape[-2]))

    meta = raster.meta.copy()
    meta.update({'height': img.shape[1],
                 'width': img.shape[2],
                 'transform': transform})

    # TODO : memory file handling function
    memory_file = MemoryFile()
    with memory_file.open(**meta) as resampled_raster:
        resampled_raster.write(img)

    return memory_file.open()


def get_intersecting_bbox(scenes):
    # Compute intersecting bounding box
    lefts = [x.bounds.left for x in scenes]
    rights = [x.bounds.right for x in scenes]
    bottoms = [x.bounds.bottom for x in scenes]
    tops = [x.bounds.top for x in scenes]
    intersection_bounds = (max(lefts), max(bottoms), min(rights), min(tops))
    bbox = geometry.box(*intersection_bounds)
    gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=rasterio.crs.CRS.from_epsg(4326))
    rasterio_friendly_coords = [json.loads(gdf.to_json())['features'][0]['geometry']]
    return rasterio_friendly_coords


def crop_to_bbox(raster, bbox):
    cropped_img, transform = mask(dataset=raster, shapes=bbox, crop=True)

    meta = raster.meta.copy()
    meta.update({'height': cropped_img.shape[1],
                 'width': cropped_img.shape[2],
                 'transform': transform})

    memory_file = MemoryFile()
    with memory_file.open(**meta) as cropped_raster:
        cropped_raster.write(cropped_img)

    return memory_file.open()


def chip_raster_into_patches(raster, patch_windows, dump_root, name_root):
    """Splits raster into patches
    """
    for idx, patch_array in enumerate(iter(iter_chip_images(raster, patch_windows))):
        patch_pil = Image.fromarray(patch_array)

        # Export chip images
        dump_path = os.path.join(dump_root, name_root + f'_{idx}_.png')
        os.makedirs(dump_root, exist_ok=True)
        with open(dump_path, 'wb') as f:
            patch_pil.save(f, format='PNG', subsampling=0, quality=100)


def retrieve_dates_from_aws_structure(directory):
    dates = []
    for year in os.listdir(directory):
        year_directory = os.path.join(directory, year)
        for month in os.listdir(year_directory):
            month_directory = os.path.join(year_directory, month)
            for day in os.listdir(month_directory):
                date = '-'.join([year, month, day])
                dates += [date]
    return dates


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'arguments: {args}')

    # Run generation
    main(args)
