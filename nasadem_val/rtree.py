"""
Source: https://github.com/gboeing/urban-data-science/blob/master/19-Spatial-Analysis-and-Cartography/rtree-spatial-indexing.ipynb
"""

from shapely.ops import unary_union
from shapely.geometry import LineString
import pandas as pd
import geopandas as gpd
import numpy as np


def quadrat_cut_geometry(geometry, quadrat_width, min_num=3, buffer_amount=1e-9):
    """
    Split a Polygon or MultiPolygon up into sub-polygons of a specified size,
    using quadrats.
    Parameters
    ----------
    geometry : shapely Polygon or MultiPolygon
        the geometry to split up into smaller sub-polygons
    quadrat_width : numeric
        the linear width of the quadrats with which to cut up the geometry (in
        the units the geometry is in)
    min_num : int
        the minimum number of linear quadrat lines (e.g., min_num=3 would
        produce a quadrat grid of 4 squares)
    buffer_amount : numeric
        buffer the quadrat grid lines by quadrat_width times buffer_amount
    Returns
    -------
    shapely MultiPolygon
    """

    # create n evenly spaced points between the min and max x and y bounds
    west, south, east, north = geometry.bounds
    x_num = int(np.ceil((east-west) / quadrat_width)) + 1
    y_num = int(np.ceil((north-south) / quadrat_width)) + 1
    x_points = np.linspace(west, east, num=max(x_num, min_num))
    y_points = np.linspace(south, north, num=max(y_num, min_num))

    # create a quadrat grid of lines at each of the evenly spaced points
    vertical_lines = [LineString([(x, y_points[0]), (x, y_points[-1])]) for x in x_points]
    horizont_lines = [LineString([(x_points[0], y), (x_points[-1], y)]) for y in y_points]
    lines = vertical_lines + horizont_lines

    # buffer each line to distance of the quadrat width divided by 1 billion,
    # take their union, then cut geometry into pieces by these quadrats
    buffer_size = quadrat_width * buffer_amount
    lines_buffered = [line.buffer(buffer_size) for line in lines]
    quadrats = unary_union(lines_buffered)
    multipoly = geometry.difference(quadrats)

    return multipoly


def obtain_intersection_point_mask(df_points, mask_geometry, ncuts_of_mask=50):

    xmin, ymin, xmax, ymax = mask_geometry.bounds
    width = min(xmax - xmin, ymax - ymin) / ncuts_of_mask
    mask_geos = quadrat_cut_geometry(mask_geometry, width)

    points_within_mask = pd.DataFrame()
    sindex = df_points.geometry.sindex
    for poly in (mask_geos):
        # buffer by the <1 micron dist to account for any space lost in the quadrat cutting
        # otherwise may miss point(s) that lay directly on quadrat line
        poly = poly.buffer(1e-14).buffer(0)

        # find approximate matches with r-tree, then precise matches from those approximate ones
        possible_matches_index = list(sindex.intersection(poly.bounds))
        possible_matches = df_points.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(poly)]
        points_within_mask = points_within_mask.append(precise_matches)

    points_within_mask = gpd.GeoDataFrame(points_within_mask,
                                          geometry=points_within_mask.geometry)
    return points_within_mask
