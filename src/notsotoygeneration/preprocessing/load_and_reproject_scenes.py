"""
Runs loading of Sentinel-2, Sentinel-1 and MODIS raw scenes, merge their bands
into single raster and reprojects them on same CRS

Usage: load_and_reproject_scenes.py [--s2=<path_to_s2_scenes_list>]  [--s1=<path_to_s1_scenes_list>] [--modis=<path_to_modis_scenes_list>]

Options:
  -h --help                            Show help.
  --version                            Show version.
  --s2=<path_to_s2_scenes_list>        Path to file listing Sentinel-2 scenes to be loaded
  --s1=<path_to_s1_scenes_list>        Path to file listing Sentinel-1 scenes to be loaded
  --modis=<path_to_modis_scenes_list>  Path to file listing MODIS scenes to be loaded
"""
from docopt import docopt
from datetime import datetime
import yaml
import rasterio
from rasterio.io import MemoryFile
from rasterio import warp
from src.notsotoygeneration.io import readers, writers

CRS = rasterio.crs.CRS.from_epsg(4326)


def main(args):

    if args['s2']:
        bands_reader = readers.S2BandReader(root=args['s2_root'])
        scene_writer = writers.S2SceneWriter(root=args['s2_output'])
        with open(args['s2_scenes'], 'r') as f:
            scenes_kwargs = yaml.safe_load(f)
        load_and_reproject_optical_scenes(reader=bands_reader, writer=scene_writer, specs=scenes_kwargs)

    if args['s1']:
        bands_reader = readers.S1BandReader(root=args['s1_root'])
        scene_writer = writers.S1SceneWriter(root=args['s1_output'])
        with open(args['s1_scenes'], 'r') as f:
            scenes_kwargs = yaml.safe_load(f)
        load_and_reproject_s1_scenes(reader=bands_reader, writer=scene_writer, specs=scenes_kwargs)

    if args['modis']:
        bands_reader = readers.MODISBandReader(root=args['modis_root'])
        scene_writer = writers.MODISSceneWriter(root=args['modis_output'])
        with open(args['modis_scenes'], 'r') as f:
            scenes_kwargs = yaml.safe_load(f)
        load_and_reproject_optical_scenes(reader=bands_reader, writer=scene_writer, specs=scenes_kwargs)


def load_and_reproject_optical_scenes(reader, writer, specs):
    """Loads scene bands rasters, stacks them together into single multiband
    raster and reprojects them at CRS global variable

    Args:
        reader (SceneReader): scene band reading utility
        writer (SceneWriter): scene writing utility
        specs (dict): specification of files to load coordinates, dates and bands
    """
    for scenes in specs['scenes']:
        for date in scenes['dates']:
            stacked_raster = stack_bands(scene_bands_reader=reader,
                                         bands=scenes['bands'],
                                         mgrs_coordinate=scenes['coordinate'],
                                         date=date)

            reproject(source_raster=stacked_raster,
                      target_crs=CRS,
                      scene_writer=writer)


def load_and_reproject_s1_scenes(reader, writer, files):
    """Loads scene bands rasters, stacks them together into single multiband
    raster and reprojects them at CRS global variable

    Args:
        reader (SceneReader): scene band reading utility
        writer (SceneWriter): scene writing utility
        files (dict): specification of Sentinel-2 files to load in MGRS and date
    """
    bands = [{'orbit': 'ascending', 'polarization': 'VH'},
             {'orbit': 'ascending', 'polarization': 'VV'},
             {'orbit': 'descending', 'polarization': 'VH'},
             {'orbit': 'descending', 'polarization': 'VV'}]

    paired_dates = map_ascending_to_descending(files['scenes'])

    for date in files['date']:
        stacked_raster = stack_bands(scene_bands_reader=reader,
                                     bands=bands,
                                     modis_coordinate=(18, 4),
                                     date=date)

        reproject(source_raster=stacked_raster,
                  target_crs=CRS,
                  scene_writer=writer)


def stack_bands(scene_bands_reader, bands, **kwargs):
    # Extract meta data from first band - assumes same for all bands
    kwargs.update({'band': bands[0]})
    meta = get_meta(scene_bands_reader=scene_bands_reader, **kwargs)
    meta.update({'count': len(bands)})

    # Create temporary in-memory file
    memory_file = MemoryFile()

    # Write new scene containing all bands from directory specified in kwargs
    with memory_file.open(**meta) as target_raster:
        for idx, band in enumerate(bands):
            kwargs.update({'band': band})
            with scene_bands_reader(**kwargs) as source_raster:
                target_raster.write_band(idx + 1, source_raster.read(1))
    return memory_file.open()


def get_meta(scene_bands_reader, **kwargs):
    with scene_bands_reader(**kwargs) as raster:
        meta = raster.meta
    return meta


def reproject(source_raster, target_crs, scene_writer, **kwargs):
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
    with scene_writer(meta=reprojected_meta, **kwargs) as reprojected_raster:
        reprojected_raster.write_band(band_indices, reprojected_img)

    return reprojected_raster


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


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'arguments: {args}')

    # Run generation
    main(args, cfg)
