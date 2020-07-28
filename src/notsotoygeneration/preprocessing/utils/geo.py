import warnings
from typing import Union
from typeguard import typechecked
import numpy as np
from geopandas import GeoDataFrame as GDF
import shapely
from shapely.geometry import Polygon
import rasterio.crs
import geopandas as gpd


@typechecked
def buffer_zero(ingeo: Union[GDF]) -> Union[GDF]:
    """
    Make invalid polygons (due to self-intersection) valid by buffering with 0.
    """
    if not all(ingeo.geometry.is_valid):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _p.buffer(0))
        return ingeo
    else:
        return ingeo


@typechecked
def close_holes(ingeo: Union[GDF, Polygon]) -> Union[GDF, Polygon]:
    """
    Close polygon holes by limitation to the exterior ring.
    """
    def _close_holes(poly: Polygon):
        if hasattr(poly, 'interiors'):
            return Polygon(list(poly.exterior.coords))
        else:
            return poly

    ingeo.geometry = ingeo.geometry.apply(_close_holes)
    return ingeo


@typechecked
def explode_mp(df: GDF) -> GDF:
    """
    Explode all multi-polygon geometries in a geodataframe into individual polygon geometries.
    Adds exploded polygons as rows at the end of the geodataframe and resets its index.
    """
    outdf = df[df.geom_type == 'Polygon']

    df_mp = df[df.geom_type == 'MultiPolygon']
    for idx, row in df_mp.iterrows():
        df_temp = gpd.GeoDataFrame(columns=df_mp.columns)
        df_temp = df_temp.append([row] * len(row.geometry), ignore_index=True)
        df_temp.geometry = list(row.geometry)
        outdf = outdf.append(df_temp, ignore_index=True)

    outdf.reset_index(drop=True, inplace=True)
    return outdf


@typechecked
def keep_biggest_poly(df: GDF) -> GDF:
    """
    Replaces MultiPolygons with the biggest polygon contained in the MultiPolygon.
    """
    row_idxs_mp = df.index[df.geometry.geom_type == 'MultiPolygon'].tolist()
    for idx in row_idxs_mp:
        mp = df.loc[idx].geometry
        max_area_poly = max(mp, key=lambda p: p.area)
        # !TODO: Maxime's comments - max_area_poly = max(poly_areas, key=lambda poly: poly.area) # pick max area polygon
        # !TODO: Maxime's comments - the above replaces both of the lines above 89-90
        df.loc[idx, 'geometry'] = max_area_poly
    return df


@typechecked
def clip(df: GDF,
         clip_poly: Polygon,
         explode_mp_: bool = False,
         keep_biggest_poly_: bool = False,
         ) -> GDF:
    """
    Filter and clip geodataframe to clipping geometry.
    The clipping geometry needs to be in the same projection as the geodataframe.
    :param df: input geodataframe
    :param clip_poly: Clipping polygon geometry, needs to be in the same crs as the input geodataframe.
    :param explode_mp_: Applies explode_mp function. Append dataframe rows for each polygon in potential
        multipolygons that were created by the intersection. Resets the dataframe index!
    :param keep_biggest_poly_: Applies keep_biggest_poly function. Replaces MultiPolygons with the biggest
        polygon contained in the MultiPolygon.
    :return: Result geodataframe.
    """
    df = df[df.geometry.intersects(clip_poly)].copy()
    df.geometry = df.geometry.apply(lambda _p: _p.intersection(clip_poly))
    # !TODO: Maxime's comments - return df

    # df = gpd.overlay(df, clip_poly, how='intersection')  # Slower.

    # !TODO: Maxime's comments - this shoud return just the clipped, should not have explosions and biggest polly

    row_idxs_mp = df.index[df.geometry.geom_type == 'MultiPolygon'].tolist()

    if not row_idxs_mp:
        return df
    elif not explode_mp_ and (not keep_biggest_poly_):
        warnings.warn(f"Warning, intersection resulted in {len(row_idxs_mp)} split multipolygons. Use "
                      f"explode_mp_=True or keep_biggest_poly_=True.")
        return df
    elif explode_mp_ and keep_biggest_poly_:
        raise ValueError('You can only use one of "explode_mp_" or "keep_biggest_poly_"!')
    elif explode_mp_:
        return explode_mp(df)
    elif keep_biggest_poly_:
        return keep_biggest_poly(df)


@typechecked
def reduce_precision(ingeo: GDF, precision: int = 3) -> GDF:
    """
    Reduces the number of after comma decimals of a shapely Polygon or geodataframe geometries.
    GeoJSON specification recommends 6 decimal places for latitude and longitude which equates to roughly 10cm of
    precision (https://github.com/perrygeo/geojson-precision).
    :param ingeo: input geodataframe.
    :param precision: number of after comma values that should remain.
    :return: Result polygon or geodataframe, same type as input.
    """
    def _reduce_precision(poly, precision):
        geojson = shapely.geometry.mapping(poly)
        try:
            geojson['coordinates'] = np.round(np.array(geojson['coordinates']), precision).tolist()
        except TypeError:
            # There is a strange polygon for which this does not work TODO! figure it out!
            pass
        poly = shapely.geometry.shape(geojson)
        if not poly.is_valid:  # Too low precision can potentially lead to invalid polygons due to line overlap effects.
            poly = poly.buffer(0)
        return poly

    ingeo.geometry = ingeo.geometry.apply(lambda _p: _reduce_precision(poly=_p, precision=precision))
    return ingeo


@typechecked
def to_pixelcoords(ingeo: GDF,
                   reference_bounds: Union[rasterio.coords.BoundingBox, tuple],
                   scale: bool = False,
                   nrows: int = None,
                   ncols: int = None
                   ) -> GDF:
    """
    Converts projected polygon coordinates to pixel coordinates of an image array.
    Subtracts point of origin, scales to pixelcoordinates.
    :param ingeo: input geodataframe or shapely Polygon.
    :param reference_bounds:  Bounding box object or tuple of reference (e.g. image chip) in format (left, bottom,
        right, top)
    :param scale: Scale the polygons to the image size/resolution. Requires image array nrows and ncols parameters.
    :param nrows: image array nrows, required for scale.
    :param ncols: image array ncols, required for scale.
    :return:Result polygon or geodataframe, same type as input.
    """
    # TODO - Maxime this function should be just to polygon
    def _to_pixelcoords(poly: Polygon, reference_bounds, scale, nrows, ncols):
        try:
            minx, miny, maxx, maxy = reference_bounds
            w_poly, h_poly = (maxx - minx, maxy - miny)
        except (TypeError, ValueError):
            raise Exception(
                f'reference_bounds argument is of type {type(reference_bounds)}, needs to be a tuple or rasterio bounding box '
                f'instance. Can be delineated from transform, nrows, ncols via rasterio.transform.reference_bounds')
        # Subtract point of origin of image bbox.
        x_coords, y_coords = poly.exterior.coords.xy
        p_origin = shapely.geometry.Polygon([[x - minx, y - miny] for x, y in zip(x_coords, y_coords)])

        if scale is False:
            return p_origin
        elif scale is True:
            if ncols is None or nrows is None:
                raise ValueError('ncols and nrows required for scale')
            x_scaler = ncols / w_poly
            y_scaler = nrows / h_poly
            return shapely.affinity.scale(p_origin, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

    ingeo.geometry = ingeo.geometry.apply(lambda _p: _to_pixelcoords(poly=_p, reference_bounds=reference_bounds,
                                                                     scale=scale, nrows=nrows, ncols=ncols))
    return ingeo


@typechecked
def invert_y_axis(ingeo: GDF,
                  reference_height: int
                  ) -> GDF:
    """
    Invert y-axis of polygon or geodataframe geometries in reference to a bounding box e.g. of an image chip.
    Usage e.g. for COCOJson format.
    :param ingeo: Input Polygon or geodataframe.
    :param reference_height: Height (in coordinates or rows) of reference object (polygon or image, e.g. image chip.
    :return: Result polygon or geodataframe, same type as input.
    """
    def _invert_y_axis(poly: Polygon = ingeo, reference_height=reference_height):
        x_coords, y_coords = poly.exterior.coords.xy
        p_inverted_y_axis = shapely.geometry.Polygon([[x, reference_height - y] for x, y in zip(x_coords, y_coords)])
        return p_inverted_y_axis

    ingeo.geometry = ingeo.geometry.apply(lambda _p: _invert_y_axis(poly=_p, reference_height=reference_height))
    return ingeo
