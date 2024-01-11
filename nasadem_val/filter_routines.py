from rasterio import features
import numpy as np
import geopandas as gpd
from rasterio.transform import rowcol
import rasterio
from rasterio.windows import transform as get_window_transform
from rasterio.windows import from_bounds as window_from_bounds
from rasterio import default_gtiff_profile
from shapely.geometry import box
from .rtree import obtain_intersection_point_mask


def rasterize_shapes_to_array(shapes: list,
                              attributes: list,
                              profile: dict,
                              all_touched=False,
                              fill_value=0):

    """
    Rasterizers a list of shapes and burns them into array with given attributes.

    Nodata from profile will be used as fill value.
    """
    out_arr = np.ones((profile['height'], profile['width']), dtype=profile['dtype'])
    out_arr = out_arr * fill_value

    # this is where we create a generator of geom, value pairs to use in rasterizing
    shapes = [(geom, value) for geom, value in zip(shapes, attributes)]
    burned = features.rasterize(shapes=shapes,
                                out=out_arr,
                                fill=fill_value,
                                transform=profile['transform'],
                                all_touched=all_touched)

    return burned


def get_raster_from_window(raster_file_path, window_bounds):
    """
    Get subset of large GIS raster.

    Assume single channel
    """
    with rasterio.open(raster_file_path) as ds:
        raster_profile_original = ds.profile

    window = window_from_bounds(*window_bounds, transform=raster_profile_original['transform'])
    window_transform = get_window_transform(window, raster_profile_original['transform'])

    with rasterio.open(raster_file_path) as ds:
        window_arr = ds.read(1, window=window)

    window_profile = get_window_profile(window, window_transform)
    return window_arr, window_profile


def get_pixel_indices(point_geo, trans):
    ind_y, ind_x = rowcol(trans, point_geo.coords[0][0], point_geo.coords[0][1])
    return ind_y, ind_x


def get_pixel_values(point_geometries, raster_data, raster_profile):

    def get_pixel_indices_partial(point_geo):
        return get_pixel_indices(point_geo, raster_profile['transform'])
    indices = point_geometries.map(get_pixel_indices_partial)
    row_indices, col_indices = zip(*indices)

    row_indices = np.clip(np.array(row_indices), 0, raster_profile['height'] - 1)
    col_indices = np.clip(np.array(col_indices), 0, raster_profile['width'] - 1)

    if isinstance(raster_data, np.ndarray):
        return raster_data[row_indices, col_indices]
    elif isinstance(raster_data, list):
        return (raster[row_indices, col_indices] for raster in raster_data)
    else:
        raise TypeError('rasterdata must be list or np.ndarray')


def get_window_profile(window, window_transform):
    profile = default_gtiff_profile.copy()
    profile['transform'] = window_transform
    profile['width'] = int(window.width)
    profile['height'] = int(window.height)
    profile['crs'] = {'init': 'epsg:4326'}
    profile['count'] = 1
    profile['nodata'] = None
    return profile


def get_pixel_values_from_raster_window(point_geometries, window_extent, raster_file_path):
    """
    Assumes all rasters are in lat, lon
    """
    if point_geometries.empty:
        return point_geometries

    window_arr, window_profile = get_raster_from_window(raster_file_path, window_extent)

    def get_pixel_indices_partial(point_geo):
        return get_pixel_indices(point_geo, window_profile['transform'])
    indices = point_geometries.map(get_pixel_indices_partial)

    row_indices, col_indices = zip(*indices)

    row_indices = np.clip(np.array(row_indices), 0, window_profile['height'] - 1)
    col_indices = np.clip(np.array(col_indices), 0, window_profile['width'] - 1)

    return window_arr[row_indices, col_indices]


def get_geometry_from_window(geometry_path, window_bounds, union_buffer=1e-5):
    df_geometry = gpd.read_file(geometry_path, bbox=tuple(window_bounds))
    geometry = df_geometry.geometry.buffer(union_buffer).unary_union.buffer(2 * union_buffer)
    return geometry


def filter_by_geometry_in_window(df, geometry_path, window_bounds, buffer=1e-5, rtree_ncuts=25):
    """
    Removes dataframe points from (non-complex) geometry with large coverage area
    """
    df_geometry = gpd.read_file(geometry_path, bbox=tuple(df.total_bounds))
    box_geo = box(*df.total_bounds)
    mask_geometry = df_geometry.geometry.intersection(box_geo).geometry.unary_union
    intersection_geometry = box_geo.difference(mask_geometry)

    df = obtain_intersection_point_mask(df,
                                        intersection_geometry,
                                        ncuts_of_mask=rtree_ncuts)
    return df
