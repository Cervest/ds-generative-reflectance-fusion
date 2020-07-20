"""
Runs loading of Sentinel-1 raw scenes, merge their bands into single raster
and reprojects them on same CRS

Usage: stack_and_reproject_s1.py --root=<raw_files_directory> --o=<output_directory> --scenes=<path_to_scenes_list>

Options:
  -h --help                            Show help.
  --version                            Show version.
  --root=<raw_files_directory>         Directory of raw Sentinel-1 files
  --o=<output_directory>               Output directory
  --scenes=<path_to_scenes_list>       Path to file listing Sentinel-1 scenes to be loaded
"""
import os
import sys
from docopt import docopt
import logging
from datetime import datetime
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
    bands_reader = readers.S1BandReader(root=args['--root'])
    scene_writer = writers.S1SceneWriter(root=args['--o'])

    # Load scenes specification file
    with open(args['--scenes'], 'r') as f:
        scenes_kwargs = yaml.safe_load(f)

    # Run loading, merging of bands and reprojection
    logging.info(f"Merging orbits and polarization of Sentinel-1 and reprojecting on CRS {CRS}")
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
    # Pair each ascending snapshot with closest descending snapshot
    paired_ascending_descending_dates = map_ascending_to_descending(specs['scenes'])
    bar = Bar(f"Merging and reprojecting", max=len(paired_ascending_descending_dates))

    for paired_date in paired_ascending_descending_dates:
        # Stack different orbits and polarizationd as different bands
        stacked_raster = stack_bands(scene_bands_reader=reader,
                                     ascending_date=paired_date['ascending'],
                                     descending_date=paired_date['descending'])

        # Reproject on unique CRS
        reproject(source_raster=stacked_raster,
                  target_crs=CRS,
                  scene_writer=writer,
                  date=paired_date['ascending'])

        bar.next()


def map_ascending_to_descending(s1_specs_scenes):
    """Ascending and descending orbits of Sentinel-1 take shots of a same view
    at different time steps. To stack these views together, we map each ascending
    time step to its closest descending time step.

    Args:
        s1_specs_scenes (dict): dictionnary of Sentinel-1 scenes to load specifications

    Returns:
        type: list[dict[str]]
    """
    # Take intersection of available sets of dates by polarization
    ascending_VH = set(s1_specs_scenes['ascending']['VH'])
    ascending_VV = set(s1_specs_scenes['ascending']['VV'])
    ascending_dates = ascending_VH.intersection(ascending_VV)

    descending_VH = set(s1_specs_scenes['descending']['VH'])
    descending_VV = set(s1_specs_scenes['descending']['VV'])
    descending_dates = descending_VH.intersection(descending_VV)

    # Pair each ascending date with closest descending date
    paired_dates = []
    for date in ascending_dates:
        date_delta_func = lambda x: abs(datetime.strptime(x, "%Y-%m-%d") - datetime.strptime(date, "%Y-%m-%d"))
        closest_descending_date = min(descending_dates, key=date_delta_func)
        paired_dates += [{'ascending': date, 'descending': closest_descending_date}]
    return paired_dates


def stack_bands(scene_bands_reader, ascending_date, descending_date):
    """Reads scenes at different bands and stacks them together into single
    raster returned in reading mode

    Args:
        scene_bands_reader (SceneReader): band reading utility for Sentinel-2
        ascending_date (str): date formatted as yyyy-mm-dd for ascending orbit
        descending_date (str): date formatted as yyyy-mm-dd for descending orbit

    Returns:
        type: rasterio.io.DatasetReader
    """
    # Extract meta data from first band - assumes same for all bands
    meta = get_meta(scene_bands_reader=scene_bands_reader,
                    date=ascending_date)
    meta.update({'count': 4})

    # Create temporary in-memory file
    memory_file = MemoryFile()

    # Write new scene containing all bands from directory specified in kwargs
    with memory_file.open(**meta) as target_raster:
        with scene_bands_reader(orbit='ascending', polarization='VH', date=ascending_date) as source_raster:
            target_raster.write_band(1, source_raster.read(1))
        with scene_bands_reader(orbit='ascending', polarization='VV', date=ascending_date) as source_raster:
            target_raster.write_band(1, source_raster.read(1))
        with scene_bands_reader(orbit='descending', polarization='VH', date=descending_date) as source_raster:
            target_raster.write_band(1, source_raster.read(1))
        with scene_bands_reader(orbit='descending', polarization='VV', date=descending_date) as source_raster:
            target_raster.write_band(1, source_raster.read(1))
    return memory_file.open()


def get_meta(scene_bands_reader, **kwargs):
    """Returns meta data of read scene given reader and loading kwargs

    Args:
        scene_bands_reader (SceneReader)
        **kwargs (dict): scene loading specification

    Returns:
        type: dict
    """
    kwargs.update({'orbit': 'ascending', 'polarization': 'VH'})
    with scene_bands_reader(**kwargs) as raster:
        meta = raster.meta
    return meta


def reproject(source_raster, target_crs, scene_writer, date):
    """Reprojects raster at specified CRS into new raster file determined
    through writer

    Args:
        source_raster (rasterio.io.DatasetReader): source raster to reproject
        target_crs (rasterio.crs.CRS)
        scene_writer (SceneWriter): writing utility for reprojected scene
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
    with scene_writer(meta=reprojected_meta, date=date, filename=None) as reprojected_raster:
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
