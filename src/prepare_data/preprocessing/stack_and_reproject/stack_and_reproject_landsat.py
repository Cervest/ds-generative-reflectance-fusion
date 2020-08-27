"""
Description:
    (1) Loads and merge raw Landsat bands rasters into single scene raster + QA raster
    (2) Reprojects rasters on specified CRS
    (3) Dumps rasters into structured directory

Usage: stack_and_reproject_landsat.py --root=<raw_scenes_directory> --o=<output_directory> --scenes_specs=<scenes_to_load>

Options:
  --root=<raw_files_directory>               Directory of raw Landsat files
  --o=<output_directory>                     Output directory
  --scenes_specs=<path_to_scenes_list>       Path to specifications YAML file about scenes to load
"""
import os
import sys
from docopt import docopt
import logging
from progress.bar import Bar
import rasterio


base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../..")
sys.path.append(base_dir)

from src.prepare_data.io import readers, writers
from src.prepare_data.preprocessing.utils import reproject_raster
from src.utils import load_yaml


def main(args):
    # Instantiate reader and writer
    bands_reader = readers.LandsatBandReader(root=args['--root'])
    scene_writer = writers.LandsatSceneWriter(root=args['--o'])

    # Load scenes specification file
    scenes_specs = load_yaml(args['--scenes_specs'])

    # Run loading, merging of bands and reprojection
    logging.info(f"Merging bands {scenes_specs['bands']} of Landsat and reprojecting on CRS:EPSG {scenes_specs['EPSG']}")
    load_stack_and_reproject_scenes(reader=bands_reader,
                                    writer=scene_writer,
                                    scenes_specs=scenes_specs)


def load_stack_and_reproject_scenes(reader, writer, scenes_specs):
    """Loads scene bands rasters, stacks them together into single multiband
    raster and reprojects them at CRS global variable

    Args:
        reader (BandReader): scene band reading utility
        writer (SceneWriter): scene writing utility
        scenes_specs (dict): specification of files to load coordinates, dates and bands
    """
    # Extract list of bands to load
    bands = scenes_specs['bands']
    crs = rasterio.crs.CRS.from_epsg(scenes_specs['EPSG'])

    # Extract list of quality map bands to load
    quality_maps = scenes_specs['quality_maps']

    for coordinate in scenes_specs['coordinates']:
        bar = Bar(f"Merging and reprojecting | Landsat Coordinate {coordinate}", max=len(scenes_specs[coordinate]['dates']))
        for date in scenes_specs[coordinate]['dates']:
            # Load multiband raster
            raster = reader.open(coordinate=coordinate,
                                 date=date,
                                 bands=bands)

            # Load QA raster
            qa_raster = reader.open(coordinate=coordinate,
                                    date=date,
                                    bands=quality_maps)

            # Reproject raster on specified CRS
            reprojected_img, reprojected_meta = reproject_raster(raster=raster, crs=crs)
            reprojected_qa_img, reprojected_qa_meta = reproject_raster(raster=qa_raster, crs=crs)

            # Write new raster and raster QA according to coordinate and date
            with writer(meta=reprojected_meta, coordinate=coordinate, date=date) as reprojected_raster:
                reprojected_raster.write(reprojected_img)

            with writer(meta=reprojected_qa_meta, coordinate=coordinate, date=date, is_quality_map=True) as reprojected_raster:
                reprojected_raster.write(reprojected_qa_img)
            bar.next()


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'arguments: {args}')

    # Run generation
    main(args)
