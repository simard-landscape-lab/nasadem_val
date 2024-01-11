import rasterio
from typing import Union, List
import numpy as np
from affine import Affine
from concurrent.futures import ThreadPoolExecutor


def get_merged_profile(profiles: List[dict],
                       res: tuple = None,
                       dtype: str = None,
                       count: int = None,
                       crs=None,
                       max_workers: int = 30,
                       nodata: Union[int, float] = None,
                       driver: str = 'GTiff') -> dict:
    """Generate Merged profiles.

    Source:
    https://gis.stackexchange.com/questions/348925/

    Parameters
    ----------
    profiles : List[dict]
        List of profiles
    res : tuple, optional
        Resolution (res_x, res_y), by default None
    dtype : str, optional
        Numpy dtype, by default None
    count : int, optional
        Rasterio count, i.e. number of channels, by default None
    crs : [type], optional
        Coordinate reference system, by default None
    max_workers : int, optional
        Max workers, by default 30
    nodata : Union[int, float], optional
        nodata for image, by default None
    driver : str, optional
        GDAL driver, by default 'GTiff'

    Returns
    -------
    dict
       Rasterio profile
    """
    first_profile = profiles[0]

    if res is None:
        res_x = first_profile['transform'].a
        res_y = -first_profile['transform'].e
    else:
        res_x, res_y = res

    if nodata is None:
        nodata = first_profile['nodata']

    if dtype is None:
        dtype = first_profile['dtype']

    if count is None:
        count = first_profile['count']

    if crs is None:
        crs = first_profile['crs']

    # Extent of all inputs
    # scan input files
    def get_bounds(profile):
        height = profile['height']
        width = profile['width']
        transform = profile['transform']
        west, south, east, north = rasterio.transform.array_bounds(height,
                                                                   width,
                                                                   transform)
        return (west, east), (south, north)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        xs_temp, ys_temp = list(zip(*list((executor.map(get_bounds,
                                                        profiles)
                                           ))))

    xs = [x for xx in xs_temp for x in xx]
    ys = [y for yy in ys_temp for y in yy]
    dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

    transform = Affine.translation(dst_w, dst_n)
    transform *= Affine.scale(res_x, -res_y)

    # Compute output array shape. We guarantee it will cover the output
    # bounds completely. Assume resolution
    width = int(np.ceil((dst_e - dst_w) / res_x))
    height = int(np.ceil((dst_n - dst_s) / res_y))

    # Adjust bounds to fit
    # dst_e, dst_s = transform * (output_width, output_height)

    dest_profile = {
                    "driver": 'GTiff',
                    "height": height,
                    "width": width,
                    "count": count,
                    "dtype": dtype,
                    "crs": crs,
                    "transform": transform,
                    "nodata": nodata,
                    }
    return dest_profile
