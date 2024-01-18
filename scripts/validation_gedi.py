# -*- coding: utf-8 -*-
"""
DEM Validation vs. GEDI Python Script

Performs validation of DEM tiles from NASADEM, SRTM V3, and GLO-30 with
GEDI GeoJSON data prepared using the other code in this package.

Authors: Michael Denbina, Charles Marshak, Marc Simard

Copyright 2023 California Institute of Technology.  All rights reserved.
United States Government Sponsorship acknowledged.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from glob import glob
import os
import zipfile
import gzip
import json
import subprocess

import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal, osr
import matplotlib.pyplot as plt
import h5py

# Basemap Related Imports
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib import ticker

# Matplotlib Style Options
plt.style.use('seaborn-white')
plt.rcParams['lines.markeredgewidth'] = 3
plt.rcParams['lines.markersize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams.update({'font.size': 18})


######################
### DATA LOCATIONS ###
######################
root_path = '/Volumes/Disk/validation_results/'
dataset_batch = np.array(['NASADEM_srtm', 'SRTMV3', 'COP']) # DEM datasets to use

# strings defining output location and type of results
version_str = 'v3_2021_12_08'
mode_str = 'bareshots' # set to 'bareshots' or 'vegshots' depending on whether you want bare earth or vegetated results
figure_str = 'run1'


# File locations for data:
gedi_path = '/mnt/phh-r0c/users/cmarshak/gedi_nasadem_l2a_aggregated/'
srtmv3_path = '/mnt/phh-r0c/users/cmarshak/srtm_v03/'
tdx_path = '/mnt/phh-r0c/users/cmarshak/tdx_90m_reproj/'
cop_path = '/mnt/phh-r0c/users/cmarshak/tdx_30m_boto/tdx_glo30/'

egm96_gtx = '/mnt/phh-r0c/users/mdenbina/egm1996/egm1996.gtx' # EGM96 Geoid GTX File
egm08_gtx = '/mnt/phh-r0c/users/mdenbina/egm2008/egm2008.gtx' # EGM2008 Geoid GTX File

glims_shp_file = '/u/phh-r0c/users/mdenbina/glims/glims_polygons_buffered.shp'


##########################
### PROCESSING OPTIONS ###
##########################
overwrite_tile_stats = False # overwrite tile statistics files if already exist
skip_plots = False # skip creating plots
plot_global = True # plot global plots in addition to continent-scale plots

percentile_threshold = [1, 99] # percentile outlier thresholds (applied to raster - lidar height differences)

# List of continents to process (NASADEM tiles and lidar data must be sorted into these subfolders within the paths specified above).
continents = np.array(['SouthAmerica', 'NorthAmerica', 'Africa', 'Australia', 'Eurasia', 'Islands'])



# LOCAL FUNCTIONS #
def covariance_regression(x,y):
    """Linear regression using the covariance matrix.  Code by Guillaume
    Brigot.

    Arguments:
        x (array): the x variable to perform the regression on.
        y (array): the y variable to perform the regression on.

    Returns:
        a: the slope of the line.
        b: the y-intercept of the line.
        rsq: the r-squared value of the line fit fit (1 - SSresidual/SStotal),
            where SSresidual is the sum of the squares of the residuals, and
            SStotal is the total sum of squares.
        corr: the Pearson correlation coefficient between the data's y
            values and the fitted line.

    """
    from numpy.linalg import eig

    ind_finite = np.isfinite(x) & np.isfinite(y)
    x = x[ind_finite]
    y = y[ind_finite]

    cov_matrix = np.cov(x, y)
    w, v = eig(cov_matrix)
    ind_max = np.argmax(w)
    mean_x = np.nanmean(x)
    mean_y = np.nanmean(y)
    a, b = np.polyfit([mean_x,mean_x - v[ind_max][0]],[mean_y,mean_y + v[ind_max][1]], 1)

    SSres = np.sum(np.square(y - (a*x + b)))
    SStot = np.sum(np.square(y - mean_y))
    rsq = (1 - SSres/SStot)
    corr = np.corrcoef((a*x + b), y)[0, 1]

    return a, b, rsq, corr


def density(x, y, mask=None, xlim=None, ylim=None, xname=None,
            yname=None, cmin=0, numbins=100, simline=True, fitline=True,
            showstats=True, units=None, cmap='gist_heat_r', lognorm=False,
            cov_regression=True, figsize=None, savefile=None, dpi=125,
            **kwargs):
    """Function for easy plotting of a 2D histogram (density plot) of the
    input x and y data (2D histogram).

    These plots provide a comparison of the distribution of two variables,
    often variables we would expect to be similar (e.g., radar and lidar
    forest heights).

    Arguments:
        x (array): the data to plot on the x axis.
        y (array): the data to plot on the y axis.
        mask (array): boolean array to choose which data is included in the
            plot.  Mask should be set to True for pixels to be plotted, False
            in pixels not to be plotted.
        xlim (tuple): the lower and upper x axis limits.  Default: (0, 50).
        ylim (tuple): the lower and upper y axis limits.  Default: Same as
            xlim.
        xname (str): the x axis label.
        yname (str): the y axis label.
        cmin: the minimum number of samples for a bin to be plotted on the
            histogram.  Default: 0 (all data).
        numbins (int): the number of histogram bins along each axis.  Default:
            100.
        simline (bool): True to display the line y=x.  False to not display.
            Default: False.
        fitline (bool): True to display a fitted line to the data.  False to
            not display.  Default: False.
        showstats (bool): True to display some stats (bias, std. dev., RMSE)
            in text on the plot.  False to not display.  Default: False.
        units (str): Units of the provided data, e.g., 'm' for height data.
            Default: '' (empty string).
        cmap: The colormap to use.  Default: 'gist_heat_r'.
        lognorm (bool): Set to True to use logarithmic scaling.  False for
            linear frequency scaling.  Default: False.
        cov_regression (bool): Set to True to use covariance matrix
            regression for the line fit, using the covariance_regression()
            function above.  Set to False to use standard least
            squares regression as implemented in SciPy.  Default: True.
        figsize (tuple): Tuple containing the figure size in the (x,y)
            dimensions (in inches).  Passed to the plt.figure() call.
        savefile (str): Path and filename to save the figure.  Default: None.
        dpi: DPI value for the saved plot, if savefile is specified.  Default:
            200.

    """
    if mask is None:
        mask = np.ones(x.shape, dtype='bool')

    if xlim is None:
        xlim = (np.floor(np.nanmin(x[mask])),np.ceil(np.nanmax(x[mask])))

    xlim = np.array(xlim)

    if units is None:
        units = ''

    ind = mask & np.isfinite(x) & np.isfinite(y)

    if ylim is None:
        ylim = xlim
    else:
        ylim = np.array(ylim)

    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)


    # 2D Histogram
    if lognorm == False:
        (counts, xedges, yedges, Image) = plt.hist2d(x[ind].flatten(),y[ind].flatten(),bins=(numbins,numbins),range=((xlim[0],xlim[1]),(ylim[0],ylim[1])),cmin=cmin,cmap=cmap,normed=True,**kwargs)
    else:
        from matplotlib.colors import LogNorm
        (counts, xedges, yedges, Image) = plt.hist2d(x[ind].flatten(),y[ind].flatten(),bins=(numbins,numbins),range=((xlim[0],xlim[1]),(ylim[0],ylim[1])),cmin=cmin,cmap=cmap,norm=LogNorm(),**kwargs)

    # Lines:
    if simline:
        plt.plot(xlim,xlim,'k--')

    if cov_regression:
        slope, intercept, rsq, corr = covariance_regression(x[ind].flatten(),y[ind].flatten())
    else:
        from scipy.stats import linregress
        slope, intercept, rvalue, pval, stderr = linregress(x[ind].flatten(),y[ind].flatten())
        rsq = rvalue**2


    if fitline == True:
        plt.plot(xlim,xlim*slope + intercept,'r--')
        if np.sign(intercept) == 1:
            plt.text((xlim[1]-xlim[0])*0.025 + xlim[0],(ylim[1]-ylim[0])*0.94 + ylim[0], 'y = '+str(np.round(slope,decimals=2))+'x + '+str(np.round(intercept,decimals=2)), size=14)
        else:
            plt.text((xlim[1]-xlim[0])*0.025 + xlim[0],(ylim[1]-ylim[0])*0.94 + ylim[0], 'y = '+str(np.round(slope,decimals=2))+'x - '+str(np.round(np.abs(intercept),decimals=2)), size=14)

    if (fitline == True) or (showstats == True):
        plt.text((xlim[1]-xlim[0])*0.025 + xlim[0],(ylim[1]-ylim[0])*0.86 + ylim[0], r'$r^2$: '+str(np.round(rsq,decimals=2)),size=14)


    bias = np.mean(y[ind] - x[ind])
    bias = np.round(bias, decimals=2)
    rms = np.sqrt(np.mean(np.square(y[ind] - x[ind])))
    rms = np.round(rms, decimals=2)
    stdev = np.std(y[ind] - x[ind])
    stdev = np.round(stdev, decimals=2)
    stdev_x = np.std(x[ind])
    stdev_x = np.round(stdev_x, decimals=2)
    n = len(y[ind])

    if showstats == True:
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.23 + ylim[0],'Bias: '+str(bias)+' '+units,size=14,horizontalalignment='right')
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.18 + ylim[0],'Y-X Std.Dev.: '+str(stdev)+' '+units,size=14,horizontalalignment='right')
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.13 + ylim[0],'X Std.Dev.: '+str(stdev_x)+' '+units,size=14,horizontalalignment='right')
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.08 + ylim[0],'RMSE: '+str(rms)+' '+units,size=14,horizontalalignment='right')
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.03 + ylim[0],'N: '+str(n),size=14,horizontalalignment='right')
    else:
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.03 + ylim[0],'N: '+str(n),size=14,horizontalalignment='right')

    if xname is not None:
        plt.xlabel(xname)

    if yname is not None:
        plt.ylabel(yname)

    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar()
    cbar.set_label('Frequency', rotation=90)

    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


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
    x0 = np.clip(x0, 0, data.shape[1]-1);
    x1 = np.clip(x1, 0, data.shape[1]-1);
    y0 = np.clip(y0, 0, data.shape[0]-1);
    y1 = np.clip(y1, 0, data.shape[0]-1);

    data_ll = data[ y0, x0 ] # lower left corner image values
    data_ul = data[ y1, x0 ] # upper left corner image values
    data_lr = data[ y0, x1 ] # lower right corner image values
    data_ur = data[ y1, x1 ] # upper right corner image values

    w_ll = (x1-x) * (y1-y) # weight for lower left value
    w_ul = (x1-x) * (y-y0) # weight for upper left value
    w_lr = (x-x0) * (y1-y) # weight for lower right value
    w_ur = (x-x0) * (y-y0) # weight for upper right value

    # Where the x or y coordinates are outside of the image boundaries, set one
    # of the weights to nan, so that these values are nan in the output array.
    ind = (x < 0)
    if np.any(ind):
        w_ll[ind] = np.nan

    ind = (x > data.shape[1]-1)
    if np.any(ind):
        w_ll[ind] = np.nan

    ind = (y < 0)
    if np.any(ind):
        w_ll[ind] = np.nan

    ind = (y > data.shape[0]-1)
    if np.any(ind):
        w_ll[ind] = np.nan

    intdata = w_ll*data_ll + w_ul*data_ul + w_lr*data_lr + w_ur*data_ur

    return intdata


def hist(x, mask=None, xlim=None, ylim=None, xname=None, yname=None,
         numbins=50, units=None, title=None, plotmean=False, showrms=False,
         figsize=None, norm=True, savefile=None, dpi=125):
    """Helper function for easy 1D histogram plotting.

    Arguments:
        x (array): data to plot.
        mask (array): boolean array to choose which data is included in the
            plot.  Mask should be set to True for pixels to be plotted, False
            in pixels not to be plotted.
        xlim (tuple): the lower and upper x axis limits.  Default: (0, 50).
        ylim (tuple): The lower and upper y axis limits.  Defaut: Determined
            by data.
        xname (str): the x axis label.
        numbins (int): the number of histogram bins.  Default: 50.
        units (str): Units of the provided data, e.g., 'm' for height data.
            Default: '' (empty string).
        title (str): Title for the plot.  Default: None.
        plotmean (bool): Plot the mean value as a vertical dashed line.
            Default: False.
        showrms (bool): Set to True to display the RMS value of x on the plot.
            (e.g., if the input data are height errors).  False: Do not
            display.  Default: False.
        figsize (tuple): Tuple containing the figure size in the (x,y)
            dimensions (in inches).  Passed to the plt.figure() call.
        norm (bool): Normalize histogram and show relative frequency.
            Default: True.
        savefile (str): Path and filename to save the figure.  Default: None.
        dpi: DPI value for the saved plot, if savefile is specified.  Default:
            200.

    """
    if mask is None:
        mask = np.ones(x.shape, dtype='bool')

    if xlim is None:
        xlim = (np.floor(np.nanmin(x[mask])),np.ceil(np.nanmax(x[mask])))

    xlim = np.array(xlim)

    if units is None:
        units = ''

    ind = mask & np.isfinite(x)

    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)

    ax = plt.gca()

    if norm:
        plt.hist(x[ind].flatten(), bins=numbins, range=xlim, density=True, edgecolor='black', linewidth=1.2)
    else:
        plt.hist(x[ind].flatten(), bins=numbins, range=xlim, edgecolor='black', linewidth=1.2)

    bias = np.mean(x[ind])
    bias = np.round(bias, decimals=2)
    rms = np.sqrt(np.mean(np.square(x[ind])))
    rms = np.round(rms, decimals=2)
    stdev = np.std(x[ind])
    stdev = np.round(stdev, decimals=2)
    n = len(x[ind])

    if ylim is None:
        ylim = ax.get_ylim()

    if plotmean:
        plt.plot([bias,bias],[ylim[0],ylim[1]],'--r')

    plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.94 + ylim[0],'Mean: '+str(bias)+' '+units,horizontalalignment='right')
    plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.86 + ylim[0],'Std.Dev.: '+str(stdev)+' '+units,horizontalalignment='right')

    if showrms:
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.78 + ylim[0],'RMS: '+str(rms)+' '+units,horizontalalignment='right')
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.70 + ylim[0],'N: '+str(n),horizontalalignment='right')
    else:
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.78 + ylim[0],'N: '+str(n),horizontalalignment='right')

    if xname is not None:
        plt.xlabel(xname)

    if yname is not None:
        plt.ylabel(yname)
    elif norm:
        plt.ylabel('Frequency')
    else:
        plt.ylabel('Count')

    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])

    if title is not None:
        plt.title(title)

    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


def degrees_to_meters(reference_latitude):
    """Calculate the number of meters covered by a degree of latitude and
        longitude at the given latitude.  Assuming WGS84 ellipsoid.

        Arguments:
            reference_latitude (float): The reference latitude, in degrees.

        Returns:
            xdist (float): Latitude spacing, in meters/degree.
            ydist (float): Longitude spacing, in meters/degree.

    """
    # WGS84 Ellipsoid
    a = 6378137.0
    b = 6356752.3142

    reference_latitude = np.radians(reference_latitude)

    meters_per_lat = np.pi / 180 * np.square(a) / b / np.power(1 + (np.square(a) - np.square(b))/np.square(b) * np.square(np.cos(reference_latitude)),1.5)
    meters_per_lon = np.pi / 180 * np.square(a) / b * np.cos(reference_latitude) / np.sqrt(1 + (np.square(a)-np.square(b))/np.square(b) * np.square(np.cos(reference_latitude)))

    return meters_per_lat, meters_per_lon


def read_geojson_gzip(input_zip_path: str):
    """ Function to read a GZIP GeoJSON file, from Charlie. """
    with gzip.GzipFile(input_zip_path, 'r') as file_in:
        data_gjson = json.loads(file_in.read().decode('utf-8'))
    return gpd.GeoDataFrame.from_features(data_gjson['features'],
                                          crs={'init': 'epsg:4326'})


def write_kea(file, data, output_transform, gdal_dtype=gdal.GDT_Int16, epsg=4326):
    """ Write a KEA raster using GDAL.  Used for saving NASADEM tiles to KEA. """
    driver = gdal.GetDriverByName('Kea')
    out = driver.Create(file, data.shape[1], data.shape[0], 1, gdal_dtype)
    out.SetGeoTransform(output_transform)
    out.GetRasterBand(1).WriteArray(data)
    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(epsg)
    out.SetProjection(out_srs.ExportToWkt())
    out.FlushCache()


def load_nasadem_hgt_zip(zip_file, nasadem_type='merged'):
    """ Load a NASADEM tile's HGT data from a ZIP archive. Returns hgt array,
        which is either a 3601x3601 array of int16 data type (for merged
        NASADEM), or float32 data type (for SRTM-only NASADEM). """
    zip_obj = zipfile.ZipFile(zip_file, 'r')

    for zip_info in zip_obj.filelist:
        if nasadem_type == 'merged':
            if os.path.splitext(zip_info.filename)[1] == '.hgt':
                hgt = np.fromstring(zip_obj.read(zip_info.filename), dtype='>i2').reshape((3601,3601))
                hgt = hgt.astype('int16')
        else:
            if os.path.splitext(zip_info.filename)[1] == '.hgts':
                hgt = np.fromstring(zip_obj.read(zip_info.filename), dtype='>f4').reshape((3601,3601))
                hgt = hgt.astype('float32')

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

    geo = (lon, spacing, 0.0, lat+1, 0.0, -1*spacing)

    return geo


def get_raster_at_ll_no_subset(raster, lat, lon, void=None, return_slope=False):
    """ Get values of given GDAL raster at input latitude and longitude. """
    geo = raster.GetGeoTransform()

    raster = raster.ReadAsArray().astype('float32')

    if void is not None:
        ind_void = (raster == void)
        if np.any(ind_void):
            raster[ind_void] = np.nan

    # Figure out row, col in subsetted raster.
    rows = (lat - geo[3]) / geo[5]
    cols = (lon - geo[0]) / geo[1]

    interp_data = bilinear_interpolate(raster, cols, rows)

    if not return_slope:
        return interp_data
    else:
        meters_per_lat, meters_per_lon = degrees_to_meters(geo[3] - 0.5)
        deg_spacing = geo[1]
        lat_spacing = deg_spacing * meters_per_lat
        lon_spacing = deg_spacing * meters_per_lon
        (lat_grad, lon_grad) = np.gradient(raster, lat_spacing, lon_spacing)
        slope = np.degrees(np.arctan(np.sqrt(lat_grad**2 + lon_grad**2)))
        interp_slope = bilinear_interpolate(slope, cols, rows)
        return interp_data, interp_slope


def get_raster_at_ll(raster, lat, lon, void=None, return_slope=False):
    """ Get values of given GDAL raster at input latitude and longitude. """
    geo = raster.GetGeoTransform()

    lat_bounds = (np.nanmin(lat), np.nanmax(lat))
    lon_bounds = (np.nanmin(lon), np.nanmax(lon))

    xoff = ((lon_bounds[0]) - geo[0])/geo[1]
    xcount = (lon_bounds[1]-lon_bounds[0])/geo[1]

    yoff = ((lat_bounds[1]) - geo[3])/geo[5]
    ycount = (lat_bounds[0]-lat_bounds[1])/geo[5]

    xoff = int(np.clip(xoff-10, 0, raster.RasterXSize-10))
    yoff = int(np.clip(yoff-10, 0, raster.RasterYSize-10))
    xcount = int(np.clip(xcount+20, 0, raster.RasterXSize-xoff))
    ycount = int(np.clip(ycount+20, 0, raster.RasterYSize-yoff))

    raster_subset = raster.ReadAsArray(xoff, yoff, xcount, ycount).astype('float32')

    if void is not None:
        ind_void = (raster_subset == void)
        if np.any(ind_void):
            raster_subset[ind_void] = np.nan

    lat_origin = geo[3] + (yoff*geo[5])
    lon_origin = geo[0] + (xoff*geo[1])

    # Figure out row, col in subsetted raster.
    rows = (lat - lat_origin) / geo[5]
    cols = (lon - lon_origin) / geo[1]

    interp_data = bilinear_interpolate(raster_subset, cols, rows)

    if not return_slope:
        return interp_data
    else:
        meters_per_lat, meters_per_lon = degrees_to_meters(geo[3] - 0.5)
        deg_spacing = geo[1]
        lat_spacing = deg_spacing * meters_per_lat
        lon_spacing = deg_spacing * meters_per_lon
        (lat_grad, lon_grad) = np.gradient(raster_subset, lat_spacing, lon_spacing)
        slope = np.degrees(np.arctan(np.sqrt(lat_grad**2 + lon_grad**2)))
        interp_slope = bilinear_interpolate(slope, cols, rows)
        return interp_data, interp_slope


def get_nasadem_elevation_at_ll(zip_file, lat, lon, max_slope=None, return_slope=False, nasadem_type='merged'):
    """ Get NASADEM elevation values at input latitude/longitude coordinates. """
    hgt = load_nasadem_hgt_zip(zip_file, nasadem_type=nasadem_type)
    geo = get_nasadem_geotransform_from_filename(zip_file)

    lat_origin = geo[3]
    lon_origin = geo[0]

    # For each field data point, check if is within TDX tile, and if so, interpolate TDX height at that point.
    rows = (lat - lat_origin) / geo[5]
    cols = (lon - lon_origin) / geo[1]

    hgt_interp = bilinear_interpolate(hgt, cols, rows)

    if (max_slope is not None) or return_slope:
        deg_spacing = geo[1]
        lat_spacing = deg_spacing * 111321
        lon_spacing = deg_spacing * np.cos(np.radians(geo[3] - 0.5)) * 111321
        (lat_grad, lon_grad) = np.gradient(hgt, lat_spacing, lon_spacing)
        slope = np.degrees(np.arctan(np.sqrt(lat_grad**2 + lon_grad**2)))
        slope_interp = bilinear_interpolate(slope, cols, rows)
        if max_slope is not None:
            hgt_interp[slope_interp > max_slope] = np.nan

    if return_slope:
        return hgt_interp, slope_interp
    else:
        return hgt_interp


def batch_get_raster_elevation_at_ll(nasadem_file_pattern, lat, lon, dem_file, mask_file=None,
                                     hem_file=None, lsm_file=None, wam_file=None,
                                     max_hem=None, max_slope=None):
    """ Get DEM elevation values at input latitude/longitude coordinates. """
    tile_files = glob(nasadem_file_pattern) # this is only used to figure out which lat/lon values to retrieve from rasters... this is done so that stats are calculated using same continent sets as NASADEM

    if dem_file is not None:
        dem = gdal.Open(dem_file)
    else:
        dem = None

    if mask_file is not None:
        mask = gdal.Open(mask_file)
    else:
        mask = None

    if hem_file is not None:
        hem = gdal.Open(hem_file)
    else:
        hem = None

    if lsm_file is not None:
        lsm = gdal.Open(lsm_file)
    else:
        lsm = None

    if wam_file is not None:
        wam = gdal.Open(wam_file)
    else:
        wam = None

    dem_vector = np.ones(lat.shape, dtype='float32') * np.nan

    for i, tile in enumerate(tile_files):
        print('Comparing {}/{}: {}                 '.format(i+1, len(tile_files), os.path.basename(tile)), end='\r')
        geo = get_nasadem_geotransform_from_filename(tile)

        bnds = (geo[0], geo[0]+1, geo[3]-1, geo[3])
        ind_bounds = (lon >= bnds[0]) & (lon <= bnds[1]) & (lat >= bnds[2]) & (lat <= bnds[3])

        if np.any(ind_bounds):
            if dem is not None:
                dem_temp, slope_temp = get_raster_at_ll(dem, lat[ind_bounds], lon[ind_bounds], return_slope=True)
            else:
                dem_temp, slope_temp = get_nasadem_elevation_at_ll(tile, lat[ind_bounds], lon[ind_bounds], return_slope=True)

            if mask is not None:
                mask_interp = get_raster_at_ll(mask, lat[ind_bounds], lon[ind_bounds])
                ind_include = np.isclose(mask_interp, 1)
            else:
                ind_include = np.ones(len(dem_temp), dtype='bool')

            if wam is not None:
                wam_interp = get_raster_at_ll(wam, lat[ind_bounds], lon[ind_bounds])
                ind_include = ind_include & (np.isclose(wam_interp, 1) | np.isclose(wam_interp, 129))

            if hem is not None:
                hem_interp = get_raster_at_ll(hem, lat[ind_bounds], lon[ind_bounds])
                ind_include = ind_include & (hem_interp < max_hem)

            if lsm is not None:
                lsm_interp = get_raster_at_ll(lsm, lat[ind_bounds], lon[ind_bounds])
                ind_include = ind_include & np.isclose(lsm_interp, 1)

            if max_slope is not None:
                ind_include = ind_include & (slope_temp < max_slope)

            if np.any(~ind_include):
                dem_temp[~ind_include] = np.nan

            dem_vector[ind_bounds] = dem_temp

    print('                                                            ', end='\r')
    return dem_vector


def get_egm_values_at_ll(lat, lon, egm='96'):
    """ Get values of EGM96 geoid at input latitude and longitude. """
    # Load EGM96 geoid:
    if (egm == '96') or (egm == '1996'):
        egm = gdal.Open(egm96_gtx)
    elif (egm == '08') or (egm == '2008'):
        egm = gdal.Open(egm08_gtx)

    egm_geo = egm.GetGeoTransform()
    egm_array = egm.ReadAsArray()

    egm_lat_origin = egm_geo[3]
    egm_lon_origin = egm_geo[0]

    # Figure out row, col in EGM array.
    egm_rows = (lat - egm_lat_origin) / egm_geo[5]
    lon_shifted = lon.copy()
    lon_shifted[lon_shifted < 0] = lon_shifted[lon_shifted < 0] + 360
    egm_cols = (lon_shifted - egm_lon_origin) / egm_geo[1]

    egm_interp = bilinear_interpolate(egm_array, egm_cols, egm_rows)

    return egm_interp


def batch_convert_nasadem_to_kea(file_pattern, outpath):
    """ Convert a bunch of NASADEM tiles in zip format to .kea rasters.

    To merge a bunch of .kea files which have been converted using this
    function, you can use something like:

    gdal_merge.py -of Kea -tap -o ../mosaic.kea *.kea

    """
    tile_files = glob(file_pattern)

    for i, tile in enumerate(tile_files):
        print('{}/{}: {}'.format(i+1, len(tile_files), os.path.basename(tile)))
        hgt = load_nasadem_hgt_zip(tile)
        geo = get_nasadem_geotransform_from_filename(tile)
        outfile = os.path.join(outpath,os.path.basename(tile).split('.')[0]+'.kea')
        write_kea(outfile, hgt, geo)


def get_continent_attributes(continent):
    """ Get attributes for each continent's basemap plot.  Called by plot_basemap. """
    radius = 40000.0 # this is in meters to set circle size

    if any(x in continent for x in ['SouthAmerica']):
        lon0 = -63
        lat0 = -18
        width = 9000000
        height = 9000000
        proj = 'laea'
    elif any(x in continent for x in ['NorthAmerica']):
        lon0 = -95
        lat0 = 45
        width = 8000000
        height = 7500000
        proj = 'laea'
    elif any(x in continent for x in ['Eurasia']):
        lon0 = 88
        lat0 = 36
        width = 18000000
        height = 11000000
        proj = 'lcc'
    elif any(x in continent for x in ['Africa']):
        lon0 = 20
        lat0 = -1.0
        width = 12500000
        height = 10000000
        proj = 'lcc'
    elif any(x in continent for x in ['Australia']):
        lon0 = 133
        lat0 = -27.5
        width = 4500000
        height = 4500000
        proj = 'lcc'
        radius = 20000.0
    elif any(x in continent for x in ['Global']):
        lon0 = 0.0
        lat0 = 0.0
        width = 0.0
        height = 0.0
        proj = 'cyl'
    else:
        print('Continent not found!')
        return None, None, None, None, None, None

    return lat0, lon0, radius, width, height, proj


def plot_basemap(continent, lat, lon, z, vmin=-5, vmax=5,
                       cmap='RdYlBu_r', cbar_label='NASADEM - Lidar (m)',
                       patch_size=None, title=None, ds_factor=250,
                       savefile=None):
    """ Create a basemap plot using Charlotte's code to show errors between
        NASADEM and ICESat for a given continent. """
    cmap = plt.get_cmap(cmap)

    lat0, lon0, radius, width_input, height_input, proj_input = get_continent_attributes(continent)
    if patch_size is None:
        patch_size = radius

    # Set up basemap.
    if proj_input == 'cyl': # Global Map
        m = Basemap(projection=proj_input, resolution='l',
                    llcrnrlat=-60, urcrnrlat=70,
                    llcrnrlon=-180, urcrnrlon=180)
    else:
        m = Basemap(width=width_input,height=height_input,
                    resolution='l',projection=proj_input,
                    lat_0=lat0, lon_0=lon0)

    m.drawcoastlines(linewidth=0.4, linestyle='solid', color='#737373', zorder=1)
    m.drawparallels(np.arange(-80.,81.,20.), linewidth=0.2, color='#737373')
    m.drawmeridians(np.arange(-180.,181.,20.), linewidth=0.2, color='#737373')

    x, y = m(lon, lat)

    # Downsample lidar shots to reduce clutter and computation time.
    if ds_factor > 1:
        x = x[::ds_factor]
        y = y[::ds_factor]
        z = z[::ds_factor]

        plot_type = 'circles'
    else:
        plot_type = 'pcolormesh'


    if plot_type == 'pcolormesh':
        lon_vector = np.linspace(np.nanmin(lon), np.nanmax(lon), num=int(np.nanmax(lon)-np.nanmin(lon)+1))
        lat_vector = np.linspace(np.nanmin(lat), np.nanmax(lat), num=int(np.nanmax(lat)-np.nanmin(lat)+1))

        xgrid, ygrid = np.meshgrid(lon_vector, lat_vector)
        zgrid = np.ones(xgrid.shape, dtype='float32') * np.nan

        for i in range(zgrid.shape[0]):
            for j in range(zgrid.shape[1]):
                ind = (lon == xgrid[i,j]) & (lat == ygrid[i,j])
                if np.sum(ind) == 1:
                    zgrid[i,j] = z[ind]

        xgrid, ygrid = m(xgrid-0.5, ygrid-0.5) # convert from lat/lon to map coordinates
        p = m.pcolormesh(xgrid, ygrid, zgrid, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat') # shading='gouraud'
    else:
        # Generate circular patches.
        patches = []
        for x1, y1 in zip(x, y):
            circle = Circle((x1,y1), patch_size)
            patches.append(circle)

        p = PatchCollection(patches, cmap=cmap, alpha=None)
        p.set_array(z)
        p.set_edgecolor(None)
        p.set_linewidth(0)
        p.set_clim(vmin=vmin, vmax=vmax)
        plt.gca().add_collection(p)


    cb = m.colorbar(p,location='bottom', cmap=cmap)
    tick_locator = ticker.MaxNLocator(nbins = 10)
    cb.locator = tick_locator
    cb.update_ticks()
    font_size = 10
    cb.set_label(cbar_label, fontsize=font_size)
    cb.ax.tick_params(labelsize=font_size, direction='in')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family("Helvetica")
    m.drawcoastlines(linewidth=0.4, linestyle='solid', color='#737373', zorder = 1)

    #if title is not None:
        # plt.title(title, size=18, color='#525252', y=1.01)   ### MARC: my python does not like this something with tex

    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight',dpi=300)

    return


def write_raster(file, data, output_transform, oformat='Kea', epsg=4326, dtype=gdal.GDT_Float32, nodataval=None):
    """ Write a KEA raster using GDAL. """
    print('Saving raster: {}'.format(file))
    driver = gdal.GetDriverByName(oformat)
    out = driver.Create(file, data.shape[1], data.shape[0], 1, dtype)
    out.SetGeoTransform(output_transform)
    out.GetRasterBand(1).WriteArray(data)
    if nodataval is not None:
        out.GetRasterBand(1).SetNoDataValue(nodataval)
    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(epsg)
    out.SetProjection(out_srs.ExportToWkt())
    out = None

    # Generate overviews.
    cmd = 'gdaladdo -r average {} 8 16 32 64 128 256'.format(file)
    print(cmd)
    print(subprocess.getoutput(cmd))

    return


def create_geotiff_stats(tile_clat, tile_clon, tile_stats, save_basename):
    """ Create GeoTIFF rasters showing the tile-level statistics (bias, RMSE,
        number of samples, and MAE). """
    UL_lat = np.nanmax(tile_clat) + 0.5
    UL_lon = np.nanmin(tile_clon) - 0.5
    raster_rows = int(UL_lat - (np.nanmin(tile_clat) - 0.5))
    raster_cols = int((np.nanmax(tile_clon) + 0.5) - UL_lon)
    raster_shape = (4, raster_rows, raster_cols) # 4 rasters: bias, std. dev., rmse, and number of samples
    raster = np.ones(raster_shape, dtype='float32') * np.nan

    for i in range(len(tile_clat)):
        row = int(UL_lat - (tile_clat[i] + 0.5))
        col = int((tile_clon[i] - 0.5) - UL_lon)

        raster[0, row, col] = tile_stats[i, 0] # bias
        raster[1, row, col] = tile_stats[i, 1] # stdev
        raster[2, row, col] = tile_stats[i, 2] # rmse
        raster[3, row, col] = tile_stats[i, 3] # num


    output_transform = (UL_lon, 1, 0, UL_lat, 0, -1)

    bias_outfile = save_basename+'_bias.tif'
    write_raster(bias_outfile, raster[0], output_transform, oformat='GTiff')

    stdev_outfile = save_basename+'_stddev.tif'
    write_raster(stdev_outfile, raster[1], output_transform, oformat='GTiff')

    rmse_outfile = save_basename+'_rmse.tif'
    write_raster(rmse_outfile, raster[2], output_transform, oformat='GTiff')

    num_outfile = save_basename+'_num.tif'
    write_raster(num_outfile, raster[3], output_transform, oformat='GTiff')

    return


def create_shp_stats(tile_clat, tile_clon, tile_stats, output_shpfile):
    df = pd.DataFrame(
        {'Center Latitude': tile_clat,
         'Center Longitude': tile_clon,
         'Bias': tile_stats[:, 0],
         'Std. Dev.': tile_stats[:, 1],
         'RMSE': tile_stats[:,2],
         'Samples': tile_stats[:,3]})
    gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df['Center Longitude'], df['Center Latitude']))

    gdf.to_file(output_shpfile)

    return


def get_tile_level_stats(nasadem_file_pattern, dem_dataset, gedi_path, forested_flag=False, max_slope_dem=5.0,
                         min_forest_cover=0, max_forest_cover=10,
                         min_h_canopy=0, max_h_canopy=5,
                         smart_orbit_filter=False, min_tile_samples=50,
                         nasadem_type='merged'):
    """ Calculate tile-level statistics between NASADEM and lidar.
        Filters pixels based on forest cover, NASADEM slope, canopy height,
        ground terrain uncertainty, etc., using provided filtering thresholds.
    """
    tile_clat = np.array([], dtype='float32')
    tile_clon = np.array([], dtype='float32')
    num_stats = 9
    tile_stats = np.zeros((0, num_stats), dtype='float32')

    diff_vector = np.array([], dtype='float32')
    diff_ground_vector = np.array([], dtype='float32')
    rh50_vector = np.array([], dtype='float32')
    rh75_vector = np.array([], dtype='float32')
    rh98_vector = np.array([], dtype='float32')

    nasadem_glob = glob(nasadem_file_pattern)

    bad_points_path = os.path.join(outpath, continent+'/outlier_point_shp/') # path for shapefiles with outlier points only
    if not os.path.isdir(bad_points_path):
        os.makedirs(bad_points_path)


    for i, file in enumerate(nasadem_glob):
        print('Processing tile #{}/{} ({})...'.format(i+1, len(nasadem_glob), os.path.basename(file)))

        nasadem_root_name = os.path.basename(file).split('.')[0]
        nasadem_geo = get_nasadem_geotransform_from_filename(file)

        gedi_geojson_gzip_file = os.path.join(gedi_path, nasadem_root_name+'.geojson.gzip')

        if os.path.isfile(gedi_geojson_gzip_file):
            gedi = read_geojson_gzip(gedi_geojson_gzip_file)

            gedi_lon = np.array(gedi['lon_lowestmode'], dtype='float64')
            gedi_lat = np.array(gedi['lat_lowestmode'], dtype='float64')
            gedi_ground = np.array(gedi['elev_lowestmode'], dtype='float32')
            gedi_rh50 = np.array(gedi['rh_50'], dtype='float32')
            gedi_rh75 = np.array(gedi['rh_75'], dtype='float32')
            gedi_rh98 = np.array(gedi['rh_98'], dtype='float32')
            gedi_hansen_forest_cover = np.array(gedi['hansen_treecover_2000'], dtype='float32')
            gedi_hansen_gain = np.array(gedi['hansen_gain'], dtype='float32')
            gedi_hansen_loss = np.array(gedi['hansen_lossyear'], dtype='float32')
            gedi_landsat_forest_cover = np.array(gedi['landsat_treecover'], dtype='float32')
            gedi_degrade_flag = np.array(gedi['degrade_flag'], dtype='float32')
            gedi_quality_flag = np.array(gedi['quality_flag'], dtype='float32')
            gedi_num_detectedmodes = np.array(gedi['num_detectedmodes'], dtype='float32')
            gedi_glims = np.array(gedi['glims_buffered'], dtype='float32')
            gedi_pekel = np.array(gedi['peckel_occurrence'], dtype='float32')
            gedi_orbit = np.array(gedi['orbit_name'])

            ind_nan_degrade = ~np.isfinite(gedi_degrade_flag)
            if np.any(ind_nan_degrade):
                gedi_degrade_flag[ind_nan_degrade] = 1
            gedi_degrade_flag = gedi_degrade_flag.astype('int32')

            ind_nan_quality = ~np.isfinite(gedi_quality_flag)
            if np.any(ind_nan_quality):
                gedi_quality_flag[ind_nan_quality] = 0
            gedi_quality_flag = gedi_quality_flag.astype('int32')

            ind_nan_detectedmodes = ~np.isfinite(gedi_num_detectedmodes)
            if np.any(ind_nan_detectedmodes):
                gedi_num_detectedmodes[ind_nan_detectedmodes] = 0
            gedi_num_detectedmodes = gedi_num_detectedmodes.astype('int32')

            # check if there's no data in this geojson file, in which case, skip it
            if (len(gedi_lon) <= 1) or (len(gedi_lat) <= 1) or (len(gedi_ground) <= 1) or (len(gedi_rh50) <= 1):
                continue

            if np.all(np.isnan(gedi_lon) | np.isnan(gedi_lat) | np.isnan(gedi_ground)):
                continue

            if 'NASADEM' in dem_dataset:
                dem_hgt, dem_slope = get_nasadem_elevation_at_ll(file, gedi_lat, gedi_lon, return_slope=True, nasadem_type=nasadem_type)
                if nasadem_type == 'merged':
                    dem_egm = get_egm_values_at_ll(gedi_lat, gedi_lon)
                    dem_hgt += dem_egm # convert NASADEM from EGM96 to WGS84 vertical datum
            elif dem_dataset == 'SRTMV3':
                srtm_file = os.path.join(srtmv3_path+continent,nasadem_root_name+'.hgt.zip')
                if os.path.isfile(srtm_file):
                    dem_hgt, dem_slope = get_nasadem_elevation_at_ll(srtm_file, gedi_lat, gedi_lon, return_slope=True)
                else:
                    continue
                dem_egm = get_egm_values_at_ll(gedi_lat, gedi_lon)
                dem_hgt += dem_egm # convert SRTMV3 from EGM96 to WGS84 vertical datum
            elif dem_dataset == 'TDX':
                tdx_file = os.path.join(tdx_path+continent,nasadem_root_name+'.tif')
                if os.path.isfile(tdx_file):
                    tdx_raster = gdal.Open(tdx_file)
                    dem_hgt, dem_slope = get_raster_at_ll_no_subset(tdx_raster, gedi_lat, gedi_lon, return_slope=True)
                else:
                    print(' -- No TDX File Found for "{}", skipping this tile...'.format(nasadem_root_name))
                    continue
            elif dem_dataset == 'COP':
                cop_root_name = nasadem_root_name.upper()
                cop_lat = cop_root_name[0:3]
                cop_lon = cop_root_name[3:]
                cop_file = os.path.join(cop_path, 'Copernicus_DSM_COG_10_{}_00_{}_00_DEM.tif'.format(cop_lat, cop_lon))
                if os.path.isfile(cop_file):
                    cop_raster = gdal.Open(cop_file)
                    dem_hgt, dem_slope = get_raster_at_ll_no_subset(cop_raster, gedi_lat, gedi_lon, return_slope=True)
                    dem_egm = get_egm_values_at_ll(gedi_lat, gedi_lon, egm='08') # COP-30 uses EGM2008 as vertical datum, convert to WGS-84
                    dem_hgt += dem_egm
                else:
                    print(' -- No COP-30 File Found for "{}", skipping this tile...'.format(nasadem_root_name))
                    continue
            else:
                print('DEM type "{}" not understood!'.format(dem_dataset))
                break

            # Compare NASADEM to ground for bare areas, RH50 for forested areas:
            if not forested_flag:
                gedi_hgt = gedi_ground
            else:
                gedi_hgt = gedi_ground + gedi_rh50

            diff = dem_hgt - gedi_hgt
            diff_ground = dem_hgt - gedi_ground


            ind_keep = np.ones_like(gedi_lat, dtype='bool')

            # Mask using terrain slope, canopy height, and other parameters...
            if forested_flag:
                ind_keep = (ind_keep
                            & (dem_hgt > -999)
                            & (dem_slope < max_slope_dem)
                            & (gedi_rh98 > min_h_canopy)
                            & (gedi_hansen_forest_cover <= max_forest_cover)
                            & (gedi_hansen_forest_cover >= min_forest_cover)
                            & (gedi_landsat_forest_cover <= max_forest_cover)
                            & (gedi_landsat_forest_cover >= min_forest_cover)
                            & (gedi_hansen_gain == 0)
                            & (gedi_hansen_loss == 0)
                            & (gedi_glims == 0)
                            & (gedi_degrade_flag == 0)
                            & (gedi_quality_flag == 1)
                            & (gedi_num_detectedmodes > 1)
                            & (gedi_pekel < 5))
            else:
                ind_keep = (ind_keep
                            & (dem_hgt > -999)
                            & (dem_slope < max_slope_dem)
                            & (gedi_rh98 < max_h_canopy)
                            & (gedi_hansen_forest_cover <= max_forest_cover)
                            & (gedi_hansen_forest_cover >= min_forest_cover)
                            & (gedi_landsat_forest_cover <= max_forest_cover)
                            & (gedi_landsat_forest_cover >= min_forest_cover)
                            & (gedi_hansen_gain == 0)
                            & (gedi_glims == 0)
                            & (gedi_degrade_flag == 0)
                            & (gedi_quality_flag == 1)
                            & (gedi_num_detectedmodes == 1)
                            & (gedi_pekel < 5))


            if smart_orbit_filter:
                # Any orbits with RMSE > mean GEDI canopy height?
                unique_orbits = np.unique(gedi_orbit)
                orbits_eligible_for_filter = np.array([], dtype=unique_orbits.dtype)

                for num, orbit in enumerate(unique_orbits):
                    ind_orbit = (gedi_orbit == orbit)
                    if np.any(ind_keep & ind_orbit):
                        orbit_rmse = np.sqrt(np.nanmean(diff[ind_keep & ind_orbit]**2))
                        if forested_flag and orbit_rmse > np.nanmedian(gedi_rh98[ind_keep]):
                            orbits_eligible_for_filter = np.append(orbits_eligible_for_filter, orbit)
                        elif not forested_flag and orbit_rmse > 10:
                            orbits_eligible_for_filter = np.append(orbits_eligible_for_filter, orbit)

                if len(orbits_eligible_for_filter) > 0:
                    original_rmse = np.sqrt(np.nanmean(diff[ind_keep]**2))

                    rmse_without_orbit = np.zeros(len(orbits_eligible_for_filter), dtype='float32')
                    for num, orbit in enumerate(orbits_eligible_for_filter):
                        ind_not_orbit = (gedi_orbit != orbit)
                        if np.any(ind_keep & ind_not_orbit):
                            rmse_without_orbit[num] = np.sqrt(np.nanmean(diff[ind_keep & ind_not_orbit]**2))
                        else:
                            rmse_without_orbit[num] = np.inf

                    # Only filter an orbit if it improves the RMSE.
                    ind_improved = rmse_without_orbit < original_rmse
                    orbits_eligible_for_filter = orbits_eligible_for_filter[ind_improved]
                    rmse_without_orbit = rmse_without_orbit[ind_improved]
                    anything_filtered = False

                    try:
                        num_shots_before_filtering = np.sum(ind_keep)
                        print(' -- RMSE of Tile Before Orbit Filter: {:.2f} ({:d} Shots)'.format(np.sqrt(np.nanmean(diff[ind_keep]**2)), num_shots_before_filtering))

                        # Remove orbits starting with the worst orbit (judged as the orbit which improves the RMSE the most when removed), but only remove an orbit if there will still be min_tile_samples shots left in the tile.
                        ind_rmse_order = np.argsort(rmse_without_orbit)
                        for orbit_to_remove in orbits_eligible_for_filter[ind_rmse_order]:
                            ind_not_orbit = (gedi_orbit != orbit_to_remove)
                            num_shots_after_filtering = np.sum(ind_keep & ind_not_orbit)
                            if np.sum(ind_keep & ind_not_orbit) >= min_tile_samples:
                                ind_keep = ind_keep & ind_not_orbit
                                print(' -- Filtered Orbit: "{}"'.format(orbit_to_remove))
                                anything_filtered = True
                            else:
                                print(' -- Could not filter orbit: "{}", not enough samples remaining ({} samples before filtering, would have {} after filtering).'.format(orbit_to_remove, num_shots_before_filtering, num_shots_after_filtering))

                        if anything_filtered:
                            print(' -- RMSE of Tile After Orbit Filter: {:.2f} ({:d} Shots)'.format(np.sqrt(np.nanmean(diff[ind_keep]**2)), num_shots_after_filtering))
                    except:
                        print(' -- Exception: Wasn\'t able to filter by orbit!')


            # Filter outliers using height difference percentiles (for bare ground analysis).
            # Also exclude unrealistic elevation values or high latitudes (outside of latitude range of SRTM coverage).
            # This is a little different than the ICESat-1 analysis, which excluded percentiles at the continent (rather than tile) level.
            # But due to the volume of data, it is hard to implement that for GEDI or ICESat-2.  But I may want to revisit this (though I'm not sure if it makes a large difference--probably not?).
            if not forested_flag and np.any(ind_keep):
                ind_outlier = (gedi_hgt < -1000) | (gedi_lat > 60) | (diff < np.nanpercentile(diff[ind_keep], percentile_threshold[0])) | (diff > np.nanpercentile(diff[ind_keep], percentile_threshold[1]))
            else:
                ind_outlier = (gedi_hgt < -1000) | (gedi_lat > 60) | (np.abs(diff) > 1000)
            ind_keep = ind_keep & (~ind_outlier)


            # Calculate statistics for this tile.
            tile_clat = np.append(tile_clat, nasadem_geo[3] - 0.5)
            tile_clon = np.append(tile_clon, nasadem_geo[0] + 0.5)
            tile_stats = np.append(tile_stats, np.zeros((1,num_stats), dtype='float32'), axis=0)

            tile_stats[-1, 0] = np.nanmean(diff[ind_keep])
            tile_stats[-1, 1] = np.nanstd(diff[ind_keep])
            tile_stats[-1, 2] = np.sqrt(np.nanmean(diff[ind_keep]**2))
            tile_stats[-1, 3] = np.sum(ind_keep)
            tile_stats[-1, 4] = np.nanmean(np.abs(diff[ind_keep]))
            tile_stats[-1, 5] = np.nanmean(diff_ground[ind_keep])
            tile_stats[-1, 6] = np.nanmean(gedi_rh50[ind_keep])
            tile_stats[-1, 7] = np.nanmean(gedi_rh98[ind_keep])
            tile_stats[-1, 8] = np.nanmean(gedi_rh98[ind_keep]/gedi_rh50[ind_keep])

            diff_vector = np.append(diff_vector, diff[ind_keep])
            diff_ground_vector = np.append(diff_ground_vector, diff_ground[ind_keep])
            rh50_vector = np.append(rh50_vector, gedi_rh50[ind_keep])
            rh75_vector = np.append(rh75_vector, gedi_rh75[ind_keep])
            rh98_vector = np.append(rh98_vector, gedi_rh98[ind_keep])
    
    return tile_clat, tile_clon, tile_stats, diff_vector, diff_ground_vector, rh50_vector, rh75_vector, rh98_vector



###################
### MAIN SCRIPT ###
###################
for dataset_str in dataset_batch:
    outpath = os.path.join(root_path, '{}_validation_gedi_{}_{}/'.format(dataset_str, mode_str, version_str))

    figurepath = os.path.join(outpath,'figures_{}/'.format(figure_str))

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if not os.path.exists(figurepath):
        os.makedirs(figurepath)

    # Bare shots or forested?
    if 'bareshots' == mode_str:
        forested_flag = False
        forested_str = 'Unforested'

        min_tile_samples = 1000
        max_slope_dem = 5.0 # maximum dem slope of 5 degrees for bare areas
        min_hansen_fc = -999
        max_hansen_fc = 10 # only include pixels with forest cover less than this amount
        min_h_canopy = -999
        max_h_canopy = 5.0 # only lidar points with canopy height less than this value will be included
        smart_orbit_filter = True

        diff_bounds_map = [-5, 5]
        diff_bounds_hist = [-10, 10]
        bias_bounds_map = [-3, 3]
        bias_bounds_hist = [-5, 5]
        rmse_bounds_map = [0, 3]
        rmse_bounds_hist = [0, 3]
    elif 'vegshots' == mode_str:
        forested_flag = True
        forested_str = 'Forested'

        min_tile_samples = 100
        max_slope_dem = 5.0 # no slope threshold for forested analysis # ACTUALLY: testing slope threshold for vegetated analysis
        min_hansen_fc = 30
        max_hansen_fc = np.inf
        min_h_canopy = 3.0
        max_h_canopy = np.inf
        smart_orbit_filter = True

        diff_bounds_map = [-10, 10]
        diff_bounds_hist = [-20, 20]
        bias_bounds_map = [-10, 10]
        bias_bounds_hist = [-10, 10]
        rmse_bounds_map = [0, 8]
        rmse_bounds_hist = [0, 10]


    # Calculate tile-level statistics.
    tile_clat_global = None
    tile_clon_global = None
    tile_stats_global = None

    continent_vector_global = np.array([], dtype='int32')
    diff_vector_global = np.array([], dtype='float32')
    diff_ground_vector_global = np.array([], dtype='float32')
    rh50_vector_global = np.array([], dtype='float32')
    rh75_vector_global = np.array([], dtype='float32')
    rh98_vector_global = np.array([], dtype='float32')

    for i, continent in enumerate(continents):
        print('Calculating tile-level statistics of {} vs. GEDI for {}...'.format(dataset_str, continent))

        if dataset_str == 'NASADEM_srtm':
            nasadem_type = 'srtm'
            file_pattern = os.path.join(root_path, '/nasadem/daac/'+continent+'/hgt_srtmOnly/*.hgts.zip')
        else:
            nasadem_type = 'merged'
            file_pattern = os.path.join(root_path, 'nasadem/daac/'+continent+'/hgt_merge/*.hgt.zip')
        
        # Files to store tile-level statistics:
        tile_clat_file = os.path.join(outpath, continent+'_'+mode_str+'_tile_clat.npy')
        tile_clon_file = os.path.join(outpath, continent+'_'+mode_str+'_tile_clon.npy')
        tile_stats_file = os.path.join(outpath, continent+'_'+mode_str+'_tile_stats.npy')

        if os.path.isfile(tile_clat_file) and os.path.isfile(tile_clon_file) and os.path.isfile(tile_stats_file) and not overwrite_tile_stats:
            tile_clat = np.load(tile_clat_file)
            tile_clon = np.load(tile_clon_file)
            tile_stats = np.load(tile_stats_file)

            if forested_flag:
                h5_veg_continent_output_file = os.path.join(outpath, continent+'_vegetation_analysis_output.h5')
                with h5py.File(h5_veg_continent_output_file, 'r') as hf_veg:
                    diff_vector = hf_veg['diff'][:]
                    diff_ground_vector = hf_veg['ground_diff'][:]
                    rh50_vector = hf_veg['rh50'][:]
                    rh75_vector = hf_veg['rh75'][:]
                    rh98_vector = hf_veg['rh98'][:]

                    continent_vector_global = np.append(continent_vector_global, np.ones(len(diff_vector))*i)
                    diff_vector_global = np.append(diff_vector_global, diff_vector)
                    diff_ground_vector_global = np.append(diff_ground_vector_global, diff_ground_vector)
                    rh50_vector_global = np.append(rh50_vector_global, rh50_vector)
                    rh75_vector_global = np.append(rh75_vector_global, rh75_vector)
                    rh98_vector_global = np.append(rh98_vector_global, rh98_vector)

        else:
            tile_clat, tile_clon, tile_stats, diff_vector, diff_ground_vector, rh50_vector, rh75_vector, rh98_vector = get_tile_level_stats(
                    file_pattern, dataset_str, os.path.join(gedi_path, continent), forested_flag,
                    max_slope_dem, min_hansen_fc, max_hansen_fc,
                    min_h_canopy=min_h_canopy, max_h_canopy=max_h_canopy,
                    smart_orbit_filter=smart_orbit_filter, min_tile_samples=min_tile_samples,
                    nasadem_type=nasadem_type)
            np.save(tile_clat_file, tile_clat)
            np.save(tile_clon_file, tile_clon)
            np.save(tile_stats_file, tile_stats)

            # Save files so we can compare vegetation bias between NASADEM compared to other datasets.
            if forested_flag:
                continent_vector_global = np.append(continent_vector_global, np.ones(len(diff_vector))*i)
                diff_vector_global = np.append(diff_vector_global, diff_vector)
                diff_ground_vector_global = np.append(diff_ground_vector_global, diff_ground_vector)
                rh50_vector_global = np.append(rh50_vector_global, rh50_vector)
                rh75_vector_global = np.append(rh75_vector_global, rh75_vector)
                rh98_vector_global = np.append(rh98_vector_global, rh98_vector)

                h5_veg_continent_output_file = os.path.join(outpath, continent+'_vegetation_analysis_output.h5')
                with h5py.File(h5_veg_continent_output_file, 'w') as hf_veg:
                    hf_veg.create_dataset('diff', data=diff_vector)
                    hf_veg.create_dataset('ground_diff', data=diff_ground_vector)
                    hf_veg.create_dataset('rh50', data=rh50_vector)
                    hf_veg.create_dataset('rh75', data=rh75_vector)
                    hf_veg.create_dataset('rh98', data=rh98_vector)

        # Append data for this continent to global arrays.
        if tile_clat_global is None:
            tile_clat_global = tile_clat.copy()
            tile_clon_global = tile_clon.copy()
            tile_stats_global = tile_stats.copy()
        else:
            tile_clat_global = np.append(tile_clat_global, tile_clat)
            tile_clon_global = np.append(tile_clon_global, tile_clon)
            tile_stats_global = np.append(tile_stats_global, tile_stats, axis=0)

        if not skip_plots:
            # Histograms of tile-level statistics.
            ind_valid = tile_stats[:,3] > min_tile_samples
            tile_clat_filt = tile_clat[ind_valid]
            tile_clon_filt = tile_clon[ind_valid]
            tile_stats_filt = tile_stats[ind_valid, :]

            hist(tile_stats_filt[:,0], xlim=bias_bounds_hist, xname='Bias of {} Tiles vs. GEDI (m)'.format(dataset_str), numbins=40, units='m', plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_GEDI_Tiles_Bias.png'.format(continent, dataset_str)))

            hist(tile_stats_filt[:,2], xlim=rmse_bounds_hist, xname='RMSE of {} Tiles vs. GEDI (m)'.format(dataset_str), numbins=40, units='m', plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_GEDI_Tiles_RMSE.png'.format(continent, dataset_str)))

            ind_valid = tile_stats[:,3] > 10
            tile_clat_filt = tile_clat[ind_valid]
            tile_clon_filt = tile_clon[ind_valid]
            tile_stats_filt = tile_stats[ind_valid, :]
            hist(np.log10(tile_stats_filt[:,3]), xlim=(1, 5), xname='log$_{10}$'+' of Number of GEDI Samples Within Each {} Tile (m)'.format(dataset_str), numbins=40, plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_GEDI_Tiles_Num.png'.format(continent, dataset_str)))
            plt.close('all')


            # Basemap of Raster-Lidar Shots
            if continent != 'Islands':
                if continent == 'NorthAmerica':
                    continent_display = 'North America'
                elif continent == 'SouthAmerica':
                    continent_display = 'South America'
                else:
                    continent_display = continent

                # Basemaps showing tile-level statistics vs. ICESat.
                saveroot_basemap_tile_stats = os.path.join(figurepath,'{}_Basemap_{}_minus_GEDI_Tiles'.format(continent, dataset_str))
                patch_size = 50000 # radius of circle used to plot each tile

                ind_valid = tile_stats[:,3] > min_tile_samples
                tile_clat_filt = tile_clat[ind_valid]
                tile_clon_filt = tile_clon[ind_valid]
                tile_stats_filt = tile_stats[ind_valid, :]

                plot_basemap(continent, tile_clat_filt, tile_clon_filt, tile_stats_filt[:,0], vmin=bias_bounds_map[0], vmax=bias_bounds_map[1],
                                   cmap='RdYlBu_r', cbar_label='Bias (m)', patch_size=patch_size,
                                   #title='Bias of {} Tiles vs. ICESat-2 \n for {} Areas of {}'.format(dataset_str, forested_str, continent_display),
                                   ds_factor=1, savefile=saveroot_basemap_tile_stats+'_Bias.png')
                plt.close('all')

                plot_basemap(continent, tile_clat_filt, tile_clon_filt, tile_stats_filt[:,2], vmin=rmse_bounds_map[0], vmax=rmse_bounds_map[1],
                                   cmap='viridis', cbar_label='RMSE (m)', patch_size=patch_size,
                                   #title='RMSE of {} Tiles vs. ICESat-2 \n for {} Areas of {}'.format(dataset_str, forested_str, continent_display),
                                   ds_factor=1, savefile=saveroot_basemap_tile_stats+'_RMSE.png')
                plt.close('all')

                ind_valid = tile_stats[:,3] > 10
                tile_clat_filt = tile_clat[ind_valid]
                tile_clon_filt = tile_clon[ind_valid]
                tile_stats_filt = tile_stats[ind_valid, :]
                plot_basemap(continent, tile_clat_filt, tile_clon_filt, np.log10(tile_stats_filt[:,3]), vmin=1, vmax=5,
                                   cmap='viridis', cbar_label=r'log$_{10}$ of Number of GEDI Samples', patch_size=patch_size,
                                   #title='Number of ICESat Samples Within Each {} Tile \n for {} Areas of {}'.format(dataset_str, forested_str, continent_display),
                                   ds_factor=1, savefile=saveroot_basemap_tile_stats+'_Num.png')
                plt.close('all')


                if forested_flag:
                    # Plot of Height Differences vs. RH100
                    density(rh98_vector, diff_ground_vector, xname='GEDI RH98 (m)', yname='{} - GEDI Ground (m)'.format(dataset_str),
                            units='m', lognorm=False, xlim=[0, 60], ylim=[-10, 50], fitline=True, showstats=False, simline=False,
                            savefile=os.path.join(figurepath,'{}_Density_{}_Ground_Diff_vs_GEDI_RH98.png'.format(continent, dataset_str)))
                    plt.close('all')

                    density(rh98_vector, diff_ground_vector, xname='GEDI RH98 (m)', yname='{} - GEDI Ground (m)'.format(dataset_str),
                            units='m', lognorm=True, xlim=[0, 60], ylim=[-10, 50], fitline=True, showstats=False, simline=False,
                            savefile=os.path.join(figurepath,'{}_LogDensity_{}_Ground_Diff_vs_GEDI_RH98.png'.format(continent, dataset_str)))
                    plt.close('all')

                    density(rh50_vector, diff_ground_vector, xname='GEDI RH50 (m)', yname='{} - GEDI Ground (m)'.format(dataset_str),
                            units='m', lognorm=False, xlim=[0, 60], ylim=[-10, 50], fitline=True, showstats=False, simline=False,
                            savefile=os.path.join(figurepath,'{}_Density_{}_Ground_Diff_vs_GEDI_RH50.png'.format(continent, dataset_str)))
                    plt.close('all')

                    density(rh50_vector, diff_ground_vector, xname='GEDI RH50 (m)', yname='{} - GEDI Ground (m)'.format(dataset_str),
                            units='m', lognorm=True, xlim=[0, 60], ylim=[-10, 50], fitline=True, showstats=False, simline=False,
                            savefile=os.path.join(figurepath,'{}_LogDensity_{}_Ground_Diff_vs_GEDI_RH50.png'.format(continent, dataset_str)))
                    plt.close('all')


                    plot_basemap(continent, tile_clat_filt, tile_clon_filt, tile_stats_filt[:,5], vmin=0, vmax=15,
                                       cmap='viridis', cbar_label='Vegetation Bias (m)', patch_size=patch_size,
                                       title='Bias of {} Tiles vs. GEDI-2 \n for {} Areas'.format(dataset_str, forested_str),
                                       ds_factor=1, savefile=saveroot_basemap_tile_stats+'_Vegetation_Bias.png')
                    plt.close('all')

                    plot_basemap(continent, tile_clat_filt, tile_clon_filt, tile_stats_filt[:,6], vmin=0, vmax=15,
                                       cmap='viridis', cbar_label='Mean GEDI RH50 (m)', patch_size=patch_size,
                                       title='RMSE of {} Tiles vs. GEDI  \n for {} Areas'.format(dataset_str, forested_str),
                                       ds_factor=1, savefile=saveroot_basemap_tile_stats+'_MeanRH50.png')
                    plt.close('all')

                    plot_basemap(continent, tile_clat_filt, tile_clon_filt, tile_stats_filt[:,7], vmin=0, vmax=30,
                                       cmap='viridis', cbar_label='Mean GEDI RH98 (m)', patch_size=patch_size,
                                       title='RMSE of {} Tiles vs. GEDI  \n for {} Areas'.format(dataset_str, forested_str),
                                       ds_factor=1, savefile=saveroot_basemap_tile_stats+'_MeanRH98.png')
                    plt.close('all')

                    plot_basemap(continent, tile_clat_filt, tile_clon_filt, tile_stats_filt[:,8], vmin=0, vmax=5,
                                       cmap='viridis', cbar_label='Mean GEDI RH98/RH50 (m)', patch_size=patch_size,
                                       ds_factor=1, savefile=saveroot_basemap_tile_stats+'_MeanRH98RH50Ratio.png')
                    plt.close('all')

                    hist(tile_stats_filt[:,5], xlim=(0, 30), xname='Vegetation Bias of {} Tiles vs. GEDI Ground (m)'.format(dataset_str), numbins=40, units='m', plotmean=False,
                                    showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_vs_GEDI_Vegetation_Bias.png'.format(continent, dataset_str)))

                    hist(tile_stats_filt[:,6], xlim=(0, 30), xname='Mean GEDI RH50 in NASADEM Tile (m)'.format(dataset_str), numbins=40, units='m', plotmean=False,
                                    showrms=False, savefile=os.path.join(figurepath,'{}_Hist_Mean_GEDI_RH50.png'.format(continent)))

                    hist(tile_stats_filt[:,7], xlim=(0, 60), xname='Mean GEDI RH98 in NASADEM Tile (m)'.format(dataset_str), numbins=40, units='m', plotmean=False,
                                    showrms=False, savefile=os.path.join(figurepath,'{}_Hist_Mean_GEDI_RH98.png'.format(continent)))

                    hist(tile_stats_filt[:,8], xlim=(0, 5), xname='Mean GEDI RH98/RH50 Ratio in NASADEM Tile (m)'.format(dataset_str), numbins=40, units='', plotmean=False,
                                    showrms=False, savefile=os.path.join(figurepath,'{}_Hist_Mean_GEDI_RH98RH50_Ratio.png'.format(continent)))

                    plt.close('all')



    if plot_global and not skip_plots:
        print('Plotting {} vs. GEDI Global Stats...'.format(dataset_str, continent))
        # Global Maps
        continent = 'Global'


        h5_veg_output_file = os.path.join(outpath, 'vegetation_analysis_output.h5')
        if forested_flag and os.path.isfile(h5_veg_output_file) and not overwrite_tile_stats:
            # Load vegetation data from h5 file.
            with h5py.File(h5_veg_output_file, 'r') as hf_veg:
                continent_vector_global = hf_veg['continent_id'][:]
                diff_vector_global = hf_veg['diff'][:]
                diff_ground_vector_global = hf_veg['ground_diff'][:]
                rh50_vector_global = hf_veg['rh50'][:]
                rh75_vector_global = hf_veg['rh75'][:]
                rh98_vector_global = hf_veg['rh98'][:]
        elif forested_flag:
            # Save files so we can compare vegetation bias between NASADEM compared to other datasets.
            with h5py.File(h5_veg_output_file, 'w') as hf_veg:
                hf_veg.create_dataset('continent_id', data=continent_vector_global)
                hf_veg.create_dataset('diff', data=diff_vector_global)
                hf_veg.create_dataset('ground_diff', data=diff_ground_vector_global)
                hf_veg.create_dataset('rh50', data=rh50_vector_global)
                hf_veg.create_dataset('rh75', data=rh75_vector_global)
                hf_veg.create_dataset('rh98', data=rh98_vector_global)
        else:
            print('Not performing vegetation analysis...')


        # Histograms of tile-level statistics for all continents.

        # For bias and RMSE, only show tiles with > min number of samples.
        ind_valid = (tile_stats_global[:,3] > min_tile_samples) & np.isfinite(tile_stats_global[:,2]) & np.isfinite(tile_stats_global[:,1])
        tile_clat_global_filt = tile_clat_global[ind_valid]
        tile_clon_global_filt = tile_clon_global[ind_valid]
        tile_stats_global_filt = tile_stats_global[ind_valid, :]

        # For number of samples within each tile, show more wider range of values.
        ind_valid = (tile_stats_global[:,3] > 10) & np.isfinite(tile_stats_global[:,2]) & np.isfinite(tile_stats_global[:,1])
        tile_clat_global_all = tile_clat_global[ind_valid]
        tile_clon_global_all = tile_clon_global[ind_valid]
        tile_stats_global_all = tile_stats_global[ind_valid, :]

        hist(tile_stats_global_filt[:,0], xlim=bias_bounds_hist, xname='Bias of {} Tiles vs. GEDI (m)'.format(dataset_str), numbins=40, units='m', plotmean=False,
                        showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_GEDI_Tiles_Bias.png'.format(continent, dataset_str)))

        hist(tile_stats_global_filt[:,2], xlim=rmse_bounds_hist, xname='RMSE of {} Tiles vs. GEDI (m)'.format(dataset_str), numbins=40, units='m', plotmean=False,
                        showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_GEDI_Tiles_RMSE.png'.format(continent, dataset_str)))

        hist(np.log10(tile_stats_global_all[:,3]), xlim=(1, 5), xname='log$_{10}$'+' of Number of GEDI Samples Within {} Tile'.format(dataset_str), numbins=40, plotmean=False,
                        showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_GEDI_Tiles_Num.png'.format(continent, dataset_str)))
        plt.close('all')

        # Overall Statistics (Percentiles of Absolute Bias and RMSE)
        print('Mean Tile Bias: {:.2f}'.format(np.nanmean(tile_stats_global_filt[:,0])))
        print('Mean MAE: {:.2f}'.format(np.nanmean(tile_stats_global_filt[:,4])))
        print('Mean RMSE: {:.2f}'.format(np.nanmean(tile_stats_global_filt[:,2])))

        print('10th Percentile of Global {} vs. ICESat-2 Bias: {:.2f}'.format(dataset_str, np.nanpercentile((tile_stats_global_filt[:,0]), 10)))
        print('90th Percentile of Global {} vs. ICESat-2 Bias: {:.2f}'.format(dataset_str, np.nanpercentile((tile_stats_global_filt[:,0]), 90)))

        print('50th Percentile of Global {} vs. ICESat MAE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,4], 50)))
        print('80th Percentile of Global {} vs. ICESat MAE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,4], 80)))
        print('90th Percentile of Global {} vs. ICESat MAE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,4], 90)))
        print('99th Percentile of Global {} vs. ICESat MAE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,4], 99)))

        print('50th Percentile of Global {} vs. ICESat RMSE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,2], 50)))
        print('80th Percentile of Global {} vs. ICESat RMSE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,2], 80)))
        print('90th Percentile of Global {} vs. ICESat RMSE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,2], 90)))
        print('99th Percentile of Global {} vs. ICESat RMSE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,2], 99)))

        print('')

        # Basemaps showing tile-level statistics vs. lidar.
        saveroot_basemap_tile_stats = os.path.join(figurepath,'{}_Basemap_{}_minus_GEDI_Tiles'.format(continent, dataset_str))

        plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,0], vmin=bias_bounds_map[0], vmax=bias_bounds_map[1],
                           cmap='RdYlBu_r', cbar_label='Bias (m)', patch_size=patch_size,
                           title='Bias of {} Tiles vs. GEDI-2 \n for {} Areas'.format(dataset_str, forested_str),
                           ds_factor=1, savefile=saveroot_basemap_tile_stats+'_Bias.png')
        plt.close('all')

        plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,2], vmin=rmse_bounds_map[0], vmax=rmse_bounds_map[1],
                           cmap='viridis', cbar_label='RMSE (m)', patch_size=patch_size,
                           title='RMSE of {} Tiles vs. GEDI  \n for {} Areas'.format(dataset_str, forested_str),
                           ds_factor=1, savefile=saveroot_basemap_tile_stats+'_RMSE.png')
        plt.close('all')

        plot_basemap(continent, tile_clat_global_all, tile_clon_global_all, np.log10(tile_stats_global_all[:,3]), vmin=1, vmax=5,
                           cmap='viridis', cbar_label=r'log$_{10}$ of Number of GEDI Samples', patch_size=patch_size,
                           title='Number of GEDI Samples Within Each Tile \n for {} Areas'.format(dataset_str, forested_str),
                           ds_factor=1, savefile=saveroot_basemap_tile_stats+'_Num.png')
        plt.close('all')

        if forested_flag:
            # Density plots showed a weird fit (presumably due to outliers) for TanDEM-X
            # I tried applying same percentile-based outlier thresholds but that removed way too much data, unfortunately, when you filter veg bias indiscriminately without considering the canopy height.
            # Instead, apply the percentile-based outlier thresholds to the vegetation bias for each canopy height bin individually...
            rh100_bin_edges = np.linspace(0, 60, num=13)
            rh100_bin_centers = (rh100_bin_edges[0:-1] + rh100_bin_edges[1:]) / 2

            ind_keep = np.ones_like(diff_ground_vector_global, dtype='bool')
            for lower_bin, upper_bin in zip(rh100_bin_edges[0:-1], rh100_bin_edges[1:]):
                ind_bin = (rh98_vector_global >= lower_bin) & (rh98_vector_global <= upper_bin)

                if np.any(ind_bin):
                    ground_diff_temp = diff_ground_vector_global[ind_bin]
                    th_lower, th_upper = np.nanpercentile(ground_diff_temp, [0.1, 99.9])

                    # Clip outlier thresholds to plot axes, so that we never discard data that would have been on the y-axis.
                    if th_upper < 60:
                        th_upper = 60

                    if th_lower > -10:
                        th_lower = -10

                    ind_outlier = ind_bin & (~np.isfinite(diff_ground_vector_global) | (diff_ground_vector_global < th_lower) | (diff_ground_vector_global > th_upper))
                    ind_keep[ind_outlier] = False

            # ind_outlier = ~np.isfinite(ground_diff_all) | (ground_diff_all < np.nanpercentile(ground_diff_all, percentile_threshold[0])) | (ground_diff_all > np.nanpercentile(ground_diff_all, percentile_threshold[1]))
            # ind_keep = ~ind_outlier
            if np.any(ind_keep):
                continent_vector_global = continent_vector_global[ind_keep]
                diff_vector_global = diff_vector_global[ind_keep]
                diff_ground_vector_global = diff_ground_vector_global[ind_keep]
                rh50_vector_global = rh50_vector_global[ind_keep]
                rh75_vector_global = rh75_vector_global[ind_keep]
                rh98_vector_global = rh98_vector_global[ind_keep]

            def do_fits_for_paper(x, y):
                from scipy.stats import linregress
                fit_slope, fit_intercept, rvalue, pval, stderr = linregress(x.flatten(),y.flatten())
                fit_y = fit_slope*x + fit_intercept
                fit_ssr = np.sum((fit_y - y)**2)
                fit_sst = np.sum((y - np.nanmean(y))**2)
                fit_r2 = 1 - (fit_ssr / fit_sst)

                zifit_slope = np.nanmean(y.flatten() / x.flatten())
                zifit_y = zifit_slope*x
                zifit_ssr = np.sum((zifit_y - y)**2)
                zifit_r2 = 1 - (zifit_ssr / fit_sst)

                return fit_slope, fit_intercept, fit_r2, zifit_slope, zifit_r2


            ind_fit = ((rh98_vector_global > 3.0) & np.isfinite(rh50_vector_global)
                      & np.isfinite(rh75_vector_global) & np.isfinite(rh98_vector_global)
                      & np.isfinite(diff_ground_vector_global))

            rmse_rh50 = np.sqrt(np.nanmean(np.square(diff_ground_vector_global[ind_fit] - rh50_vector_global[ind_fit])))
            rmse_rh75 = np.sqrt(np.nanmean(np.square(diff_ground_vector_global[ind_fit] - rh75_vector_global[ind_fit])))
            rmse_rh98 = np.sqrt(np.nanmean(np.square(diff_ground_vector_global[ind_fit] - rh98_vector_global[ind_fit])))

            print('Veg. Stats for Paper ({} vs. GEDI):'.format(dataset_str))
            print('')
            print('RH98:')
            fit_slope, fit_intercept, fit_r2, zifit_slope, zifit_r2 = do_fits_for_paper(rh98_vector_global[ind_fit], diff_ground_vector_global[ind_fit])
            print('Linear Fit. Slope: {}, Intercept: {}, r2: {}'.format(fit_slope, fit_intercept, fit_r2))
            print('Linear Fit w/ Zero Intercept.  Slope: {}, r2: {}'.format(zifit_slope, zifit_r2))
            print('RMSE vs. RH98: {}'.format(rmse_rh98))
            print('')
            print('RH75:')
            fit_slope, fit_intercept, fit_r2, zifit_slope, zifit_r2 = do_fits_for_paper(rh75_vector_global[ind_fit], diff_ground_vector_global[ind_fit])
            print('Linear Fit. Slope: {}, Intercept: {}, r2: {}'.format(fit_slope, fit_intercept, fit_r2))
            print('Linear Fit w/ Zero Intercept.  Slope: {}, r2: {}'.format(zifit_slope, zifit_r2))
            print('RMSE vs. RH75: {}'.format(rmse_rh75))
            print('')
            print('RH50')
            fit_slope, fit_intercept, fit_r2, zifit_slope, zifit_r2 = do_fits_for_paper(rh50_vector_global[ind_fit], diff_ground_vector_global[ind_fit])
            print('Linear Fit. Slope: {}, Intercept: {}, r2: {}'.format(fit_slope, fit_intercept, fit_r2))
            print('Linear Fit w/ Zero Intercept.  Slope: {}, r2: {}'.format(zifit_slope, zifit_r2))
            print('RMSE vs. RH50: {}'.format(rmse_rh50))
            print('')



            # Plot of Height Differences vs. RH100
            density(rh98_vector_global, diff_ground_vector_global, xname='GEDI RH98 (m)', yname='{} - GEDI Ground (m)'.format(dataset_str),
                    units='m', lognorm=True, xlim=[0, 60], ylim=[-5, 60], fitline=True, showstats=False, simline=False,
                    savefile=os.path.join(figurepath,'Global_LogDensity_{}_Ground_Diff_vs_GEDI_RH98.png'.format(dataset_str)))
            plt.close('all')

            density(rh98_vector_global, diff_ground_vector_global, xname='GEDI RH98 (m)', yname='{} - GEDI Ground (m)'.format(dataset_str),
                    units='m', lognorm=False, xlim=[0, 60], ylim=[-5, 60], fitline=True, showstats=False, simline=False,
                    savefile=os.path.join(figurepath,'Global_Density_{}_Ground_Diff_vs_GEDI_RH98.png'.format(dataset_str)))
            plt.close('all')

            plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,5], vmin=0, vmax=15,
                               cmap='viridis', cbar_label='Mean Vegetation Bias In Tile (m)', patch_size=patch_size,
                               title='Bias of {} Tiles vs. GEDI-2 \n for {} Areas'.format(dataset_str, forested_str),
                               ds_factor=1, savefile=saveroot_basemap_tile_stats+'_Vegetation_Bias.png')
            plt.close('all')

            plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,6], vmin=0, vmax=15,
                               cmap='viridis', cbar_label='Mean GEDI RH50 In Tile (m)', patch_size=patch_size,
                               title='RMSE of {} Tiles vs. GEDI  \n for {} Areas'.format(dataset_str, forested_str),
                               ds_factor=1, savefile=saveroot_basemap_tile_stats+'_MeanRH50.png')
            plt.close('all')

            plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,7], vmin=0, vmax=30,
                               cmap='viridis', cbar_label='Mean GEDI RH98 In Tile (m)', patch_size=patch_size,
                               title='RMSE of {} Tiles vs. GEDI  \n for {} Areas'.format(dataset_str, forested_str),
                               ds_factor=1, savefile=saveroot_basemap_tile_stats+'_MeanRH98.png')
            plt.close('all')

            plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,7]/tile_stats_global_filt[:,6], vmin=0, vmax=6,
                               cmap='viridis', cbar_label='Mean GEDI RH98/RH50 Ratio In Tile (m)', patch_size=patch_size,
                               ds_factor=1, savefile=saveroot_basemap_tile_stats+'_MeanRH98RH50Ratio.png')
            plt.close('all')


            hist(tile_stats_global_filt[:,5], xlim=(0, 30), xname='Vegetation Bias of {} Tiles vs. GEDI Ground (m)'.format(dataset_str), numbins=40, units='m', plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_vs_GEDI_Vegetation_Bias.png'.format(continent, dataset_str)))

            hist(tile_stats_global_filt[:,6], xlim=(0, 30), xname='Mean GEDI RH50 (m)'.format(dataset_str), numbins=40, units='m', plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_Mean_GEDI_RH50.png'.format(continent)))

            hist(tile_stats_global_filt[:,7], xlim=(0, 60), xname='Mean GEDI RH98 (m)'.format(dataset_str), numbins=40, units='m', plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_Mean_GEDI_RH98.png'.format(continent)))

            hist(tile_stats_global_filt[:,7]/tile_stats_global_filt[:,6], xlim=(0, 6), xname='Mean GEDI RH98/RH50 Ratio (m)'.format(dataset_str), numbins=40, units='', plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_Mean_GEDI_RH98RH50_Ratio.png'.format(continent)))



        geotiff_basename = os.path.join(figurepath,'Global_{}_v_GEDI_Tile_Stats'.format(dataset_str))
        create_geotiff_stats(tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt, geotiff_basename)

        output_shpfile = os.path.join(figurepath,'Global_{}_v_GEDI_Tile_Stats.shp'.format(dataset_str))
        create_shp_stats(tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt, output_shpfile)


print(' -- validation_gedi.py Finished! --')