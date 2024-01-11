"""Author: Michael Denbina
"""

import os
import zipfile
import numpy as np
from affine import Affine
from rasterio import default_gtiff_profile


default_profile = default_gtiff_profile.copy()
default_profile['nodata'] = -10_000
default_profile['dtype'] = np.int16


def bilinear_interpolate(data, x, y):
    """Function to perform bilinear interpolation on the input array data, at
    the image coordinates given by input arguments x and y.

    Arguments
        data (array): 2D array containing raster data to interpolate.
        x (array): the X coordinate values at which to interpolate (in array
            indices, starting at zero).  Note that X refers to the second
            dimension of data (e.g., the columns).
        y (array): the Y coordinate values at which to interpolate (in array
            indices, starting at zero).  Note that Y refers to the first
            dimension of data (e.g., the rows).

    Returns:
        intdata (array): The 2D interpolated array, with same dimensions as
            x and y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Get lower and upper bounds for each pixel.
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Clip the image coordinates to the size of the input data.
    x0 = np.clip(x0, 0, data.shape[1]-1)
    x1 = np.clip(x1, 0, data.shape[1]-1)
    y0 = np.clip(y0, 0, data.shape[0]-1)
    y1 = np.clip(y1, 0, data.shape[0]-1)

    data_ll = data[y0, x0]  # lower left corner image values
    data_ul = data[y1, x0]  # upper left corner image values
    data_lr = data[y0, x1]  # lower right corner image values
    data_ur = data[y1, x1]  # upper right corner image values

    w_ll = (x1-x) * (y1-y)  # weight for lower left value
    w_ul = (x1-x) * (y-y0)  # weight for upper left value
    w_lr = (x-x0) * (y1-y)  # weight for lower right value
    w_ur = (x-x0) * (y-y0)  # weight for upper right value

    # Where the x or y coordinates are outside of the image boundaries, set one
    # of the weights to nan, so that these values are nan in the output array.
    w_ll[np.less(x, 0)] = np.nan
    w_ll[np.greater(x, data.shape[1]-1)] = np.nan
    w_ll[np.less(y, 0)] = np.nan
    w_ll[np.greater(y, data.shape[0]-1)] = np.nan

    intdata = w_ll*data_ll + w_ul*data_ul + w_lr*data_lr + w_ur*data_ur

    return intdata


def load_nasadem_hgt_zip(zip_file):
    """ Load a NASADEM tile's HGT data from a ZIP archive. Returns hgt array,
        which is a 3601x3601 array of int16 data type. """
    zip_obj = zipfile.ZipFile(zip_file, 'r')

    for zip_info in zip_obj.filelist:
        if '.hgt' == os.path.splitext(zip_info.filename)[1]:
            hgt = np.fromstring(zip_obj.read(zip_info.filename), dtype='>i2').reshape((3601, 3601))
    hgt = hgt.astype('int16')
    return hgt


def get_nasadem_geotransform_from_filename(zip_file):
    """ From a given NASADEM tile filename, return a GDAL-style GeoTransform()
        tuple. """
    tile_name = os.path.basename(zip_file).split('.')[0].upper()

    if 'N' in tile_name:
        lat_str = 'N'
    elif 'S' in tile_name:
        lat_str = 'S'
    else:
        print('Cannot parse NASADEM tile filename!')
        return None

    if 'E' in tile_name:
        lon_str = 'E'
    elif 'W' in tile_name:
        lon_str = 'W'
    else:
        print('Cannot parse NASADEM tile filename!')
        return None

    lon = int(tile_name.split(lon_str)[1])
    lat = int(tile_name.split(lon_str)[0][1:])

    if lat_str == 'S':
        lat *= -1

    if lon_str == 'W':
        lon *= -1

    spacing = 0.0002777777777777778

    transform = (lon, spacing, 0.0, lat+1, 0.0, -1*spacing)

    return transform


def get_profile_from_zip_file(zip_file):
    profile = default_profile.copy()

    transform = get_nasadem_geotransform_from_filename(zip_file)
    transform = Affine.from_gdal(*transform)
    profile['transform'] = transform

    profile['width'] = profile['height'] = 3601
    profile['crs'] = {'init': 'epsg:4326'}
    profile['count'] = 1
    return profile


def get_nasadem_data(zip_file):
    profile = get_profile_from_zip_file(zip_file)
    hgt = load_nasadem_hgt_zip(zip_file)
    return hgt, profile


def get_dem_slope(dem, trans):
    """Author: Michael Denbina; adapted for rasterio"""
    deg_spacing = trans[0]
    lat_spacing = deg_spacing * 111321
    lon_spacing = deg_spacing * np.cos(np.radians(trans[5] - 0.5)) * 111321
    (lat_grad, lon_grad) = np.gradient(dem, lat_spacing, lon_spacing)
    slope = np.degrees(np.arctan(np.sqrt(lat_grad**2 + lon_grad**2)))
    return slope
