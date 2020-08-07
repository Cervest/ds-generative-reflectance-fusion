import numpy as np
import json
import geopandas as gpd
import rasterio
from rasterio import warp
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.enums import Resampling


def in_memory_raster(array, meta):
    """Encapsulates image array and metadata as raster instance in reading mode
    stored in memory i.e. no actual file written

    Args:
        array (np.ndarray): raster image content
        meta (dict): raster metadata

    Returns:
        type: rasterio.io.DatasetReader
    """
    memory_file = MemoryFile()
    with memory_file.open(**meta) as raster:
        raster.write(array)
    return memory_file.open()


def reproject_raster(raster, crs, as_raster=False):
    """Reprojects raster at specified CRS into new raster dataset

    Args:
        raster (rasterio.io.DatasetReader): source raster to reproject
        crs (rasterio.crs.CRS): target crs for reprojection

    Returns:
        type: np.ndarray, dict
    """
    # Cast raster as Band to apply reprojection transform
    band_indices = list(range(1, 1 + raster.count))
    source_as_band = rasterio.band(ds=raster, bidx=band_indices)

    # Compute reprojected array and associated transform
    reprojected_img, transform = warp.reproject(source=source_as_band,
                                                dst_crs=crs)

    # Update metadata accordingly
    reprojected_meta = raster.meta.copy()
    reprojected_meta.update({'height': reprojected_img.shape[1],
                             'width': reprojected_img.shape[2],
                             'transform': transform,
                             'crs': crs})

    # Return output in suited format
    if as_raster:
        reprojected_raster = in_memory_raster(reprojected_img, reprojected_meta)
        return reprojected_raster
    else:
        return reprojected_img, reprojected_meta


def crop_raster_to_bbox(raster, bbox, as_raster=False):
    """Crops raster to window defined by bounding box

    Args:
        raster (rasterio.io.DatasetReader): source raster to crop
        bbox (shapely.geometry.Box): cropping bounding box

        which is then converted to rasterio friendly format as :

        ```
            [{'type': 'Polygon',
              'coordinates': [[[5.642740780274462, 48.70507037789256],
                               [5.642740780274462, 49.33640635957176],
                               [4.357910769844605, 49.33640635957176],
                               [4.357910769844605, 48.70507037789256],
                               [5.642740780274462, 48.70507037789256]]]}]
        ```

    Returns:
        type: np.ndarray, dict
    """
    # Format bouding box coordinates to rasterio-friendly fashion
    gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=rasterio.crs.CRS.from_epsg(4326))
    rasterio_friendly_bbox = [json.loads(gdf.to_json())['features'][0]['geometry']]

    # Compute cropped array and associated transform
    cropped_img, transform = mask(dataset=raster, shapes=rasterio_friendly_bbox, crop=True)

    # Update metadata accordingly
    cropped_meta = raster.meta.copy()
    cropped_meta.update({'height': cropped_img.shape[1],
                         'width': cropped_img.shape[2],
                         'transform': transform})

    # Return output in suited format
    if as_raster:
        cropped_raster = in_memory_raster(cropped_img, cropped_meta)
        return cropped_raster
    else:
        return cropped_img, cropped_meta


def resample_raster(raster, resolution, as_raster=False):
    """Resamples raster to specified resolution

    Args:
        raster (rasterio.io.DatasetReader): source raster to resample
        resolution (tuple[float]): (height_resolution, width_resolution)

    Returns:
        type: np.ndarray, dict
    """
    # Compute resampling ratios
    height_resampling = raster.res[0] / resolution[0]
    width_resampling = raster.res[1] / resolution[1]

    # Compute resampled array and associated transform
    resampled_img = raster.read(out_shape=(raster.count,
                                           int(np.ceil(raster.height * height_resampling)),
                                           int(np.ceil(raster.width * width_resampling))),
                                resampling=Resampling.bilinear)
    transform = raster.transform * raster.transform.scale((raster.width / resampled_img.shape[-1]),
                                                          (raster.height / resampled_img.shape[-2]))

    # Update metadata accordingly
    resampled_meta = raster.meta.copy()
    resampled_meta.update({'height': resampled_img.shape[1],
                           'width': resampled_img.shape[2],
                           'transform': transform})

    # Return output in suited format
    if as_raster:
        resampled_raster = in_memory_raster(resampled_img, resampled_meta)
        return resampled_raster
    else:
        return resampled_img, resampled_meta
