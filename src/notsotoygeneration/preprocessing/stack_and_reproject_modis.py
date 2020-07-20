"""
Runs loading of MODIS raw scenes, merge their bands into single raster
and reprojects them on same CRS

Usage: stack_and_reproject_modis.py --root=<raw_files_directory> --o=<output_directory> --scenes=<path_to_scenes_list>

Options:
  -h --help                            Show help.
  --version                            Show version.
  --root=<raw_files_directory>         Directory of raw MODIS files
  --o=<output_directory>               Output directory
  --scenes=<path_to_scenes_list>       Path to file listing MODIS scenes to be loaded
"""
import os
import sys
from docopt import docopt
import logging
import yaml
from progress.bar import Bar
import rasterio
from rasterio.io import MemoryFile
from rasterio import warp

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
sys.path.append(base_dir)

from src.notsotoygeneration.io import readers, writers

CRS = rasterio.crs.CRS.from_epsg(4326)


def main(args):
    # Instantiate reader and writer
    bands_reader = readers.MODISBandReader(root=args['--root'])
    scene_writer = writers.MODISSceneWriter(root=args['--o'])

    # Load scenes specification file
    with open(args['--scenes'], 'r') as f:
        scenes_kwargs = yaml.safe_load(f)

    # Run loading, merging of bands and reprojection
    logging.info(f"Merging bands {scenes_kwargs['bands']} of MODIS and reprojecting on CRS {CRS}")
    load_stack_and_reproject_scenes(reader=bands_reader,
                                    writer=scene_writer,
                                    specs=scenes_kwargs)


def load_stack_and_reproject_scenes(reader, writer, specs):
    """Loads scene bands rasters, stacks them together into single multiband
    raster and reprojects them at CRS global variable

    Args:
        reader (SceneReader): scene band reading utility
        writer (SceneWriter): scene writing utility
        specs (dict): specification of files to load coordinates, dates and bands
    """
    for scenes in specs['scenes']:
        modis_coordinates = scenes['coordinate']
        dates = scenes['dates']
        bar = Bar(f"Merging and reprojecting | Coordinate {modis_coordinates}", max=len(dates))

        for date in dates:
            stacked_raster = stack_bands(scene_bands_reader=reader,
                                         bands=specs['bands'],
                                         modis_coordinate=modis_coordinates,
                                         date=date)

            reproject(source_raster=stacked_raster,
                      target_crs=CRS,
                      scene_writer=writer,
                      modis_coordinate=scenes['coordinate'],
                      date=date)

            bar.next()


def stack_bands(scene_bands_reader, bands, modis_coordinate, date):
    """Reads scenes at different bands and stacks them together into single
    raster returned in reading mode

    Args:
        scene_bands_reader (SceneReader): band reading utility for Sentinel-2
        bands (list[str]): list of bands to stack together
        modis_coordinate (tuple[int]): (horizontal tile, vertical tile)
        date (str): date formatted as yyyy-mm-dd

    Returns:
        type: rasterio.io.DatasetReader
    """
    # Extract meta data from first band - assumes same for all bands
    meta = get_meta(scene_bands_reader=scene_bands_reader,
                    modis_coordinate=modis_coordinate,
                    date=date,
                    band=bands[0])
    meta.update({'count': len(bands)})

    # Create temporary in-memory file
    memory_file = MemoryFile()

    # Write new scene containing all bands from directory specified in kwargs
    with memory_file.open(**meta) as target_raster:
        for idx, band in enumerate(bands):
            with scene_bands_reader(modis_coordinate=modis_coordinate, date=date, band=band) as source_raster:
                target_raster.write_band(idx + 1, source_raster.read(1))
    return memory_file.open()


def get_meta(scene_bands_reader, **kwargs):
    """Returns meta data of read scene given reader and loading kwargs

    Args:
        scene_bands_reader (SceneReader)
        **kwargs (dict): scene loading specification

    Returns:
        type: dict
    """
    with scene_bands_reader(**kwargs) as raster:
        meta = raster.meta
    return meta


def reproject(source_raster, target_crs, scene_writer, modis_coordinate, date):
    """Reprojects raster at specified CRS into new raster file determined
    through writer

    Args:
        source_raster (rasterio.io.DatasetReader): source raster to reproject
        target_crs (rasterio.crs.CRS)
        scene_writer (SceneWriter): writing utility for reprojected scene
        modis_coordinate (tuple[int]): (horizontal tile, vertical tile)
        date (str): date formatted as yyyy-mm-dd
    """
    # Cast raster as Band to apply reprojection transform
    band_indices = list(range(1, 1 + source_raster.count))
    source_as_band = rasterio.band(ds=source_raster, bidx=band_indices)

    # Compute reprojected array and associated transform
    reprojected_img, transform = warp.reproject(source=source_as_band,
                                                dst_crs=target_crs)

    # Update metadata accordingly
    reprojected_meta = source_raster.meta.copy()
    reprojected_meta.update({'driver': 'GTiff',
                             'height': reprojected_img.shape[1],
                             'width': reprojected_img.shape[2],
                             'transform': transform,
                             'crs': target_crs})

    # Write new reprojected raster
    with scene_writer(meta=reprojected_meta, modis_coordinate=modis_coordinate, date=date, filename=None) as reprojected_raster:
        reprojected_raster.write_band(band_indices, reprojected_img)

    return reprojected_raster


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'arguments: {args}')

    # Run generation
    main(args)
