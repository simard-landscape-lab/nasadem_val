# -*- coding: utf-8 -*-
"""
DEM Validation vs. ICESat Python Script

Performs validation of DEM tiles from NASADEM, SRTM V3, and GLO-30 with
ICESat data.

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
import subprocess
import zipfile

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
version_str = 'v7_2021_12_08'
icesat_str = 'bareshots' # set to 'bareshots' or 'vegshots' depending on whether you want bare earth or vegetated results
figure_str = 'run1'

icesat_file = os.path.join(root_path, 'data/icesat1/GLA14_634_global_{}_clean_020117_NASADEMREADY.dat'.format(icesat_str))
egm96_gtx = os.path.join(root_path, 'egm1996/egm1996.gtx')# EGM96 Geoid GTX File
egm08_gtx = os.path.join(root_path, 'egm2008/egm2008.gtx') # EGM08 Geoid GTX File
srtmv3_path = os.path.join(root_path, 'srtm_v03/')
cop_path = os.path.join(root_path, 'tdx_30m_boto/tdx_glo30/')


##########################
### PROCESSING OPTIONS ###
##########################
do_icesat_comparison = True
overwrite_npy_files = False
plot_icesat_comparison = True
overwrite_icesat_tile_stats = False
plot_global = True
convert_npy_to_csv = False

percentile_threshold = [1, 99] # percentile outlier thresholds (applied to raster - ICESat height differences)

# List of continents to process (NASADEM tiles must be sorted into these subfolders within the nasadem_path specified above).
continents = np.array(['Africa', 'Australia', 'Eurasia', 'Islands', 'NorthAmerica', 'SouthAmerica'])



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

    lat_origin = geo[3]# + (geo[5]/2)
    lon_origin = geo[0]# + (geo[1]/2)

    # For each field data point, check if is within TDX tile, and if so, interpolate TDX height at that point.
    rows = (lat - lat_origin) / geo[5]
    cols = (lon - lon_origin) / geo[1]

    hgt_interp = bilinear_interpolate(hgt, cols, rows)

    if (max_slope is not None) or return_slope:
        meters_per_lat, meters_per_lon = degrees_to_meters(geo[3] - 0.5)
        deg_spacing = geo[1]
        lat_spacing = deg_spacing * meters_per_lat
        lon_spacing = deg_spacing * meters_per_lon
        (lat_grad, lon_grad) = np.gradient(hgt, lat_spacing, lon_spacing)
        slope = np.degrees(np.arctan(np.sqrt(lat_grad**2 + lon_grad**2)))
        slope_interp = bilinear_interpolate(slope, cols, rows)
        if max_slope is not None:
            hgt_interp[slope_interp > max_slope] = np.nan

    if return_slope:
        return hgt_interp, slope_interp
    else:
        return hgt_interp


def batch_get_raster_elevation_at_ll(nasadem_file_pattern, lat, lon, dem_source=None, mask_file=None,
                                     hem_file=None, lsm_file=None, wam_file=None,
                                     max_hem=None, max_slope=None, nasadem_type='merged'):
    """ Get DEM elevation values at input latitude/longitude coordinates. """
    tile_files = glob(nasadem_file_pattern) # this is only used to figure out which lat/lon values to retrieve from rasters... this is done so that stats are calculated using same continent sets as NASADEM

    if (dem_source is not None) and os.path.isfile(dem_source):
        dem = gdal.Open(dem_source)
        dem_path = None
    elif (dem_source is not None) and os.path.isdir(dem_source): # DEM path, not single file
        dem = None
        dem_path = dem_source
    else:
        dem = None
        dem_path = None

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
            if dem is not None: # Get Elevation from DEM Raster File
                dem_temp, slope_temp = get_raster_at_ll(dem, lat[ind_bounds], lon[ind_bounds], return_slope=True)
            elif dem_path is not None: # Get Elevation from DEM Raster Path (with same filename conventions as NASADEM, e.g., SRTMV3)
                nasadem_root_name = os.path.basename(tile).split('.')[0]
                if dataset_str == 'SRTMV3':
                    srtm_file = os.path.join(dem_path, nasadem_root_name+'.hgt.zip')
                    if os.path.isfile(srtm_file):
                        dem_temp, slope_temp = get_nasadem_elevation_at_ll(srtm_file, lat[ind_bounds], lon[ind_bounds], return_slope=True)
                    else:
                        print(' -- No SRTM File Found for "{}", skipping this tile...'.format(nasadem_root_name))
                        continue
                elif dataset_str == 'TDX':
                    tdx_file = os.path.join(dem_path,nasadem_root_name+'.tif')
                    if os.path.isfile(tdx_file):
                        tdx_raster = gdal.Open(tdx_file)
                        dem_temp, slope_temp = get_raster_at_ll_no_subset(tdx_raster, lat[ind_bounds], lon[ind_bounds], return_slope=True)
                    else:
                        print(' -- No TDX File Found for "{}", skipping this tile...'.format(nasadem_root_name))
                        continue
                elif dataset_str == 'COP':
                    cop_root_name = nasadem_root_name.upper()
                    cop_lat = cop_root_name[0:3]
                    cop_lon = cop_root_name[3:]
                    cop_file = os.path.join(cop_path, 'Copernicus_DSM_COG_10_{}_00_{}_00_DEM.tif'.format(cop_lat, cop_lon))
                    if os.path.isfile(cop_file):
                        cop_raster = gdal.Open(cop_file)
                        dem_temp, slope_temp = get_raster_at_ll_no_subset(cop_raster, lat[ind_bounds], lon[ind_bounds], return_slope=True)
                    else:
                        print(' -- No COP-30 File Found for "{}", skipping this tile...'.format(nasadem_root_name))
                        continue
            else: # Get NASADEM Elevation
                dem_temp, slope_temp = get_nasadem_elevation_at_ll(tile, lat[ind_bounds], lon[ind_bounds], return_slope=True, nasadem_type=nasadem_type)

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
    """ Get attributes for each continent's basemap plot.  Called by plot_bias_basemap. """
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
                       cmap='RdYlBu_r', cbar_label='NASADEM - ICESat (m)',
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

    # Downsample ICESat shots to reduce clutter and computation time.
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


def multibinplot(x, y, cat_id, cat_label_vector, mask=None, xlim=None,
                 ylim=None, xbins=50, xname=None, yname=None, mincount=10,
                 figsize=None, savefile=None, dpi=125,
                 simline=False, show_stdev=False, plot_global=True):
    """Plot the average (and, optionally, the standard deviation) of one
    parameter as a function of another, for multiple series.

    The mean value of the y data in each x bin is plotted with a solid
    line.  The mean plus and minus the standard deviation is shown as a
    shaded region.

    Arguments:
        x (array): Data array for the x axis.
        y (array): Data array for the y axis.
        mask (array): Boolean array.  Where mask == True, the data will be
            included in the plot.  Where mask == False, the data will be
            excluded.
        xlim (tuple): X-axis limits.  Tuple of (min, max).  Default: Min and
            max of data in x.
        ylim (tuple): Y-axis limits.  Tuple of (min, max).  Default: Min and
            max of data in y.
        xbins (int): Number of bins for the x axis.  Default: 50.
        xname (str): The x-axis label.
        yname (str): The y-axis label.
        mincount (int): Minimum number of samples in a bin for the bin to
            be plotted.  Default: 10.
        figsize (tuple): Tuple containing the figure size in the (x,y)
            dimensions (in inches).  Passed to the plt.figure() call.
        savefile (str): Path and filename to save the figure.  Default: None.
        dpi: DPI value for the saved plot, if savefile is specified.  Default:
            200.
        simline (bool): Plot the y=x line.  Default: False.
        show_stdev (bool): Plot one standard deviation around the mean as a
            shaded region.  Default: False.

    """
    if mask is None:
        mask = np.ones(x.shape, dtype='bool')

    ind = mask & np.isfinite(x) & np.isfinite(y)

    if np.any(ind):
        x = x[ind]
        y = y[ind]
        cat_id = cat_id[ind]
    else:
        print('kapok.plot.binplot | No valid data in masked region.  Aborting.')
        return

    if xlim is None:
        xlim = (np.nanmin(x), np.nanmax(x))

    binwidth = (xlim[1] - xlim[0]) / xbins
    bincenters = np.linspace(xlim[0]+(binwidth/2),xlim[1]-(binwidth/2),num=xbins)

    cat_label_vector = np.insert(cat_label_vector, 0, 'Global')

    num_cats = len(cat_label_vector)

    y_avg = np.zeros((num_cats, xbins), dtype=y.dtype)
    y_std = np.zeros((num_cats, xbins), dtype=y.dtype)

    for cat in range(num_cats):
        if cat_label_vector[cat] == 'Global':
            ind_cat = np.isfinite(cat_id)
        else:
            ind_cat = (cat_id == (cat-1))

        for num in range(xbins):
            lowerbound = num*binwidth + xlim[0]
            upperbound = (num+1)*binwidth + xlim[0]

            ind = ind_cat & (x >= lowerbound) & (x < upperbound)

            if np.sum(ind) >= mincount:
                y_avg[cat, num] = np.nanmedian(y[ind])
                y_std[cat, num] = np.nanstd(y[ind])
            else:
                y_avg[cat, num] = np.nan
                y_std[cat, num] = np.nan

    if figsize is None:
        fig, ax = plt.subplots(1)
    else:
        fig, ax = plt.subplots(1, figsize=figsize)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for cat in range(num_cats):
        if cat_label_vector[cat] == 'NorthAmerica':
            label_str = 'North America'
        elif cat_label_vector[cat] == 'SouthAmerica':
            label_str = 'South America'
        elif cat_label_vector[cat] == 'Islands':
            continue
        elif (cat_label_vector[cat] == 'Global') and not plot_global:
            continue
        else:
            label_str = cat_label_vector[cat]

        if show_stdev:
            ax.fill_between(bincenters, y_avg[cat]+y_std[cat], y_avg[cat]-y_std[cat], facecolor=colors[int(cat) % len(colors)], alpha=0.1)

        plt.plot(bincenters, y_avg[cat], color=colors[int(cat) % len(colors)], linewidth=3, label=label_str)
        plt.xlim(xlim)

    if plot_global:
        plt.plot(bincenters, y_avg[0], color=colors[0], linewidth=3)

    if xname is not None:
        plt.xlabel(xname)

    if ylim is not None:
        plt.ylim(ylim)

    if yname is not None:
        plt.ylabel(yname)

    plt.legend(loc=0, frameon=True, fontsize=10)

    if not isinstance(simline, bool): # simline is slope of line to plot
        plt.plot(xlim,xlim*simline,'k--', label='y = {}x'.format(simline))
    elif isinstance(simline, bool) and simline:
        plt.plot(xlim,xlim,'k--', label='y = x')

    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


def comparisonbinplot(x, y, cat_id, cat_label_vector, dataset_label_vector,
                      xlim=None, ylim=None, xbins=50,
                      xname=None, yname=None, mincount=10,
                      figsize=None, savefile=None, dpi=125,
                      simline=False, show_stdev=False,
                      savefile_diff=None, savefile_global=None):
    """Plot the average (and, optionally, the standard deviation) of one
    parameter as a function of another, for multiple series.

    The mean value of the y data in each x bin is plotted with a solid
    line.  The mean plus and minus the standard deviation is shown as a
    shaded region.

    Arguments:
        x (array): Data array for the x axis.
        y (array): Data array for the y axis.
        mask (array): Boolean array.  Where mask == True, the data will be
            included in the plot.  Where mask == False, the data will be
            excluded.
        xlim (tuple): X-axis limits.  Tuple of (min, max).  Default: Min and
            max of data in x.
        ylim (tuple): Y-axis limits.  Tuple of (min, max).  Default: Min and
            max of data in y.
        xbins (int): Number of bins for the x axis.  Default: 50.
        xname (str): The x-axis label.
        yname (str): The y-axis label.
        mincount (int): Minimum number of samples in a bin for the bin to
            be plotted.  Default: 10.
        figsize (tuple): Tuple containing the figure size in the (x,y)
            dimensions (in inches).  Passed to the plt.figure() call.
        savefile (str): Path and filename to save the figure.  Default: None.
        dpi: DPI value for the saved plot, if savefile is specified.  Default:
            200.
        simline (bool): Plot the y=x line.  Default: False.
        show_stdev (bool): Plot one standard deviation around the mean as a
            shaded region.  Default: False.

    """
    line_styles = ['-', '--']

    num_keys = 0
    for key in x:
        num_keys += 1
        ind = np.isfinite(x[key]) & np.isfinite(y[key])

        if np.any(ind):
            x[key] = x[key][ind]
            y[key] = y[key][ind]
            cat_id[key] = cat_id[key][ind]
        else:
            print('kapok.plot.binplot | No valid data in masked region.  Aborting.')
            return

        if xlim is None:
            xlim = (np.nanmin(x[key]), np.nanmax(x[key]))

    binwidth = (xlim[1] - xlim[0]) / xbins
    bincenters = np.linspace(xlim[0]+(binwidth/2),xlim[1]-(binwidth/2),num=xbins)
    cat_label_vector = np.insert(cat_label_vector, 0, 'Global')
    num_cats = len(cat_label_vector)

    y_avg = np.zeros((num_keys, num_cats, xbins), dtype=y['NASADEM'].dtype)
    y_std = np.zeros((num_keys, num_cats, xbins), dtype=y['NASADEM'].dtype)

    for num_key, key in enumerate(dataset_label_vector):
        for cat in range(num_cats):
            if cat_label_vector[cat] == 'Global':
                ind_cat = np.isfinite(cat_id[key])
            else:
                ind_cat = (cat_id[key] == (cat-1))

            for num in range(xbins):
                lowerbound = num*binwidth + xlim[0]
                upperbound = (num+1)*binwidth + xlim[0]

                ind = ind_cat & (x[key] >= lowerbound) & (x[key] < upperbound)

                if np.sum(ind) >= mincount:
                    y_avg[num_key, cat, num] = np.nanmean(y[key][ind])
                    y_std[num_key, cat, num] = np.nanstd(y[key][ind])
                else:
                    y_avg[num_key, cat, num] = np.nan
                    y_std[num_key, cat, num] = np.nan

    if figsize is None:
        fig, ax = plt.subplots(1)
    else:
        fig, ax = plt.subplots(1, figsize=figsize)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for cat in range(num_cats):
        if cat_label_vector[cat] == 'NorthAmerica':
            continent_str = 'North America'
        elif cat_label_vector[cat] == 'SouthAmerica':
            continent_str = 'South America'
        elif cat_label_vector[cat] == 'Islands':
            continue
        else:
            continent_str = cat_label_vector[cat]

        for num_key, key in enumerate(dataset_label_vector):
            if show_stdev:
                ax.fill_between(bincenters, y_avg[num_key, cat]+y_std[num_key, cat], y_avg[num_key, cat]-y_std[num_key, cat], facecolor=colors[int(cat) % len(colors)], alpha=0.1)

            label_str = '{}, {}'.format(dataset_label_vector[key], continent_str)
            plt.plot(bincenters, y_avg[num_key, cat], line_styles[num_key], color=colors[int(cat) % len(colors)], linewidth=2, label=label_str)

        plt.xlim(xlim)

    for num_key, key in enumerate(dataset_label_vector):
        plt.plot(bincenters, y_avg[num_key, 0], line_styles[num_key], color=colors[0], linewidth=2)

    if xname is not None:
        plt.xlabel(xname)

    if ylim is not None:
        plt.ylim(ylim)

    if yname is not None:
        plt.ylabel(yname)

    plt.legend(loc=0, frameon=True, fontsize=8)

    if not isinstance(simline, bool): # simline is slope of line to plot
        plt.plot(xlim,xlim*simline,'k--', label='y = {}x'.format(simline))
    elif isinstance(simline, bool) and simline:
        plt.plot(xlim,xlim,'k--', label='y = x')

    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


    # Create a less busy plot showing global-level trends (and maybe South America and Africa only?).
    if savefile_global is not None:
        if figsize is None:
            fig, ax = plt.subplots(1)
        else:
            fig, ax = plt.subplots(1, figsize=figsize)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for cat in [0]: # specific continents only
            if cat_label_vector[cat] == 'NorthAmerica':
                continent_str = 'North America'
            elif cat_label_vector[cat] == 'SouthAmerica':
                continent_str = 'South America'
            elif cat_label_vector[cat] == 'Islands':
                continue
            else:
                continent_str = cat_label_vector[cat]

            for num_key, key in enumerate(dataset_label_vector):
                label_str = '{}'.format(dataset_label_vector[key], continent_str)
                plt.plot(bincenters, y_avg[num_key, cat], '-', color=colors[int(num_key) % len(colors)], linewidth=2, label=label_str)

            plt.xlim(xlim)

        if xname is not None:
            plt.xlabel(xname)

        if ylim is not None:
            plt.ylim(ylim)

        if yname is not None:
            plt.ylabel(yname)

        plt.legend(loc=0, frameon=True, fontsize=10)

        if not isinstance(simline, bool): # simline is slope of line to plot
            plt.plot(xlim,xlim*simline,'k--', label='y = {}x'.format(simline))
        elif isinstance(simline, bool) and simline:
            plt.plot(xlim,xlim,'k--', label='y = x')

        plt.savefig(savefile_global, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


    # Plot the difference in vegetation bias between the various datasets.
    if savefile_diff is not None:
        if figsize is None:
            fig, ax = plt.subplots(1)
        else:
            fig, ax = plt.subplots(1, figsize=figsize)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for cat in range(num_cats):
            if cat_label_vector[cat] == 'NorthAmerica':
                continent_str = 'North America'
            elif cat_label_vector[cat] == 'SouthAmerica':
                continent_str = 'South America'
            elif cat_label_vector[cat] == 'Islands':
                continue
            else:
                continent_str = cat_label_vector[cat]

            for num_key, key in enumerate(dataset_label_vector):
                if num_key > 0:
                    label_str = '{} - {}, {}'.format(dataset_label_vector[key], dataset_label_vector['NASADEM'], continent_str)
                    plt.plot(bincenters, y_avg[num_key, cat] - y_avg[0, cat], '-', color=colors[int(cat) % len(colors)], linewidth=2, label=label_str)

            plt.xlim(xlim)

        for num_key, key in enumerate(dataset_label_vector):
            if num_key > 0:
                plt.plot(bincenters, y_avg[num_key, 0] - y_avg[0, 0], '-', color=colors[0], linewidth=2)

        if xname is not None:
            plt.xlabel(xname)

        plt.ylim([-2, 8])

        plt.ylabel('Difference in Vegetation Bias (m)')

        plt.legend(loc=0, frameon=True, fontsize=10)

        plt.savefig(savefile_diff, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


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


def get_tile_level_stats(lat, lon, diff, min_samples=1, ground_diff=None, canopyhgt=None, trt=None):
    """ From vectors of latitude, longitude, and height difference, return
        NASADEM tile-level statistics (e.g., bias, std. dev., and RMSE for
        each 1 degree by 1 degree NASADEM tile). """
    # Bounds of input latitude and longitudes.
    lat_bounds = [np.floor(np.nanmin(lat)), np.ceil(np.nanmax(lat))]
    lon_bounds = [np.floor(np.nanmin(lon)), np.ceil(np.nanmax(lon))]

    # Center latitude and longitude of tiles.
    tile_clat = np.arange(lat_bounds[0]+0.5, lat_bounds[1])
    tile_clon = np.arange(lon_bounds[0]+0.5, lon_bounds[1])

    num_tiles_lat = len(tile_clat)
    num_tiles_lon = len(tile_clon)

    tile_clat = np.tile(tile_clat, (num_tiles_lon, 1)).T
    tile_clon = np.tile(tile_clon, (num_tiles_lat, 1))

    tile_clat = tile_clat.flatten()
    tile_clon = tile_clon.flatten()

    tile_stats = np.zeros((len(tile_clat), 14), dtype='float32')

    for i, clat in enumerate(tile_clat):
        print('Calculating Tile Stats {}/{}...          '.format(i+1, len(tile_clat)), end='\r')
        clon = tile_clon[i]

        ind = ((lat >= clat-0.5) & (lat <= clat+0.5) &
               (lon >= clon-0.5) & (lon <= clon+0.5))

        tile_stats[i, 3] = np.sum(ind)
        if tile_stats[i, 3] > min_samples:
            tile_stats[i, 0] = np.nanmean(diff[ind])
            tile_stats[i, 1] = np.nanstd(diff[ind])
            tile_stats[i, 2] = np.sqrt(np.nanmean(diff[ind]**2))
            tile_stats[i, 4] = np.nanmean(np.abs(diff[ind]))
            tile_stats[i, 5] = np.nanmedian(diff[ind])

            if canopyhgt is not None:
                tile_stats[i, 6] = np.nanmean(canopyhgt[ind])
                tile_stats[i, 7] = np.nanmedian(canopyhgt[ind])

            if trt is not None:
                tile_stats[i, 8] = np.nanmean(trt[ind])
                tile_stats[i, 9] = np.nanmedian(trt[ind])

            if ground_diff is not None:
                tile_stats[i, 10] = np.nanmean(ground_diff[ind])
                tile_stats[i, 11] = np.nanstd(ground_diff[ind])
                tile_stats[i, 12] = np.sqrt(np.nanmean(ground_diff[ind]**2))
                tile_stats[i, 13] = np.nanmean(np.abs(ground_diff[ind]))

    print('                                                            ', end='\r')
    return tile_clat, tile_clon, tile_stats



###################
### MAIN SCRIPT ###
###################
for dataset_str in dataset_batch:
    if dataset_str == 'NASADEM_srtm':
        dataset_disp = 'NASADEM'
    elif dataset_str == 'NASADEM_merged':
        dataset_disp = 'NASADEM (Merged)'
    elif dataset_str == 'COP':
        dataset_disp = 'Copernicus DEM'
    else:
        dataset_disp = dataset_str

    outpath = os.path.join(root_path, '{}_validation_{}_{}/'.format(dataset_str, icesat_str, version_str))

    figurepath = os.path.join(outpath,'figures/')

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if not os.path.exists(figurepath):
        os.makedirs(figurepath)


    if dataset_str == 'NASADEM' or dataset_str == 'NASADEM_srtm' or dataset_str == 'NASADEM_merged':
        dem_source = None
        dem_path = None
        mask_file = None
        hem_file = None
        lsm_file = None
        wam_file = None
        if dataset_str == 'NASADEM_srtm':
            vdatum = 'wgs84'
        else:
            vdatum = 'egm96'
    elif dataset_str == 'SRTMV3':
        dem_source = 'path'
        dem_path = srtmv3_path
        mask_file = None
        hem_file = None
        lsm_file = None
        wam_file = None
        vdatum = 'egm96'
    elif dataset_str == 'COP':
        dem_source = 'path'
        dem_path = cop_path
        mask_file = None
        hem_file = None
        lsm_file = None
        wam_file = None
        vdatum = 'egm08'


    # Bare shots or forested?
    if 'bareshots' == icesat_str:
        forested_flag = False
        forested_str = 'Unforested'
        veg_rh_ref = None

        max_slope_dem = 5.0 # maximum dem slope of 5 degrees for bare areas
        min_tile_samples = 1000

        diff_bounds_map = [-5, 5]
        diff_bounds_hist = [-10, 10]
        bias_bounds_map = [-3, 3]
        bias_bounds_hist = [-3, 3]
        rmse_bounds_map = [0, 3]
        rmse_bounds_hist = [0, 3]
    elif 'vegshots' == icesat_str:
        forested_flag = True
        forested_str = 'Forested'
        veg_rh_ref = 'RH50'

        max_slope_dem = 5.0 # no slope threshold for forested analysis # actually, turn on slope threshold of 5 degrees to test
        min_tile_samples = 100

        diff_bounds_map = [-10, 10]
        diff_bounds_hist = [-20, 20]
        bias_bounds_map = [-10, 10]
        bias_bounds_hist = [-10, 10]
        rmse_bounds_map = [0, 8]
        rmse_bounds_hist = [0, 10]


    # Output ICESAT data w/ DEM height column added, for later analysis and plotting.
    # Use a separate output file for each continent.
    if do_icesat_comparison:
        # Load ICESat data.
        icesat = pd.read_csv(icesat_file)
        icesat_array = np.array(icesat).astype('float32')
        if vdatum == 'egm96':
            icesat_egm = get_egm_values_at_ll(icesat_array[:,0], icesat_array[:,1])
        elif vdatum == 'egm08':
            icesat_egm = get_egm_values_at_ll(icesat_array[:,0], icesat_array[:,1], egm='08')
        else:
            icesat_egm = None

        for continent in continents:
            print('Comparing {} DEM vs. ICESat for {}...'.format(dataset_str, continent))
            if dataset_str == 'NASADEM_merged':
                nasadem_type = 'merged'
                file_pattern = '/mnt/phh-r0b/nasadem/daac/'+continent+'/hgt_merge/*.hgt.zip'
            elif dataset_str == 'NASADEM_srtm':
                nasadem_type = 'srtm'
                file_pattern = '/mnt/phh-r0b/nasadem/daac/'+continent+'/hgt_srtmOnly/*.hgts.zip'
            else:
                nasadem_type = 'merged'
                file_pattern = '/mnt/phh-r0b/nasadem/daac/'+continent+'/hgt_merge/*.hgt.zip'
            
            outfile = os.path.join(outpath, continent+'_'+os.path.splitext(os.path.basename(icesat_file))[0]+'.npy')
            if not os.path.isfile(outfile) or overwrite_npy_files:
                if dem_path is not None:
                    if dataset_str == 'COP':
                        dem_heights = batch_get_raster_elevation_at_ll(file_pattern, icesat_array[:,0], icesat_array[:,1], dem_source=dem_path, mask_file=mask_file, hem_file=hem_file, lsm_file=lsm_file, wam_file=wam_file, max_hem=0.5, max_slope=max_slope_dem, nasadem_type=nasadem_type)
                    else:
                        dem_path_continent = os.path.join(dem_path, continent)
                        dem_heights = batch_get_raster_elevation_at_ll(file_pattern, icesat_array[:,0], icesat_array[:,1], dem_source=dem_path_continent, mask_file=mask_file, hem_file=hem_file, lsm_file=lsm_file, wam_file=wam_file, max_hem=0.5, max_slope=max_slope_dem, nasadem_type=nasadem_type)
                else:
                    dem_heights = batch_get_raster_elevation_at_ll(file_pattern, icesat_array[:,0], icesat_array[:,1], dem_source=dem_source, mask_file=mask_file, hem_file=hem_file, lsm_file=lsm_file, wam_file=wam_file, max_hem=0.5, max_slope=max_slope_dem, nasadem_type=nasadem_type)

                ind_valid = np.isfinite(dem_heights)
                if icesat_egm is not None:
                    dem_heights = dem_heights[ind_valid] + icesat_egm[ind_valid]
                else:
                    dem_heights = dem_heights[ind_valid]
                icesat_sub = np.concatenate((icesat_array[ind_valid, :], dem_heights[:, np.newaxis]), axis=1)
                np.save(outfile, icesat_sub)


    # Generate validation plots for NASADEM vs. ICESat.
    if plot_icesat_comparison:
        tile_clat_global = None
        tile_clon_global = None
        tile_stats_global = None

        if forested_flag: # vectors to store all points to generate later plots for vegetation analysis
            continent_id_all = np.array([])
            diff_all = np.array([])
            ground_diff_all = np.array([])
            rh50_all = np.array([])
            rh75_all = np.array([])
            rh100_all = np.array([])
            trt_all = np.array([])

        for i, continent in enumerate(continents):
            print('Plotting {} vs. ICESat for {}...'.format(dataset_str, continent))
            npyfile = os.path.join(outpath, continent+'_'+os.path.splitext(os.path.basename(icesat_file))[0]+'.npy')
            icesat_str_disp = 'ICESat Ground'
            icesat_sub = np.load(npyfile)

            if forested_flag:
                ground_diff = icesat_sub[:,-1] - icesat_sub[:,2] # ground elevation difference
                if veg_rh_ref is not None:
                    icesat_str_disp = 'ICESat {}'.format(veg_rh_ref)
                    if veg_rh_ref == 'RH50':
                        diff = icesat_sub[:,-1] - (icesat_sub[:,2] + icesat_sub[:,3]) # difference between raster and RH50
                    elif veg_rh_ref == 'RH75':
                        diff = icesat_sub[:,-1] - (icesat_sub[:,2] + icesat_sub[:,4]) # difference between raster and RH75
                    elif veg_rh_ref == 'RH100':
                        diff = icesat_sub[:,-1] - (icesat_sub[:,2] + icesat_sub[:,5]) # difference between raster and RH100
                    else:
                        print('Error: RH value {} not recognized.'.format(veg_rh_ref))
                else:
                    icesat_str_disp = 'ICESat Ground'
                    diff = icesat_sub[:,-1] - icesat_sub[:,2] # ground elevation difference
            else:
                icesat_str_disp = 'ICESat RH50'
                diff = icesat_sub[:,-1] - icesat_sub[:,2] # in bareshots file, only RH50 elevation is given -- only vegshots file has different RH metrics to compare against


            # Remove outliers and ICESat shots with lat above 60N and ICESat shots with impossible elevation values.
            if 'bareshots' in icesat_file:
                # ind_outlier = (icesat_sub[:,2] < -1000) | (icesat_sub[:,0] > 60) | (diff > (np.nanmean(diff)+np.nanstd(diff)*num_std_threshold)) | (diff < (np.nanmean(diff)-np.nanstd(diff)*num_std_threshold))
                ind_outlier = (icesat_sub[:,2] < -1000) | (icesat_sub[:,0] > 60) | (diff < np.nanpercentile(diff, percentile_threshold[0])) | (diff > np.nanpercentile(diff, percentile_threshold[1]))
            else:
                ind_outlier = (icesat_sub[:,2] < -1000) | (icesat_sub[:,0] > 60) | (np.abs(diff) > 1000)
            if np.any(~ind_outlier):
                icesat_sub = icesat_sub[~ind_outlier, :]
                diff = diff[~ind_outlier]
                if forested_flag:
                    ground_diff = ground_diff[~ind_outlier]


            # Histogram of Height Differences
            hist(diff, xlim=diff_bounds_hist, xname='{} - {} Elevation (m)'.format(dataset_str, icesat_str_disp), numbins=80, units='m', plotmean=False,
                            showrms=True, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_ICESat.png'.format(continent, dataset_str)))

            # 2D Histogram of Elevations
            density(icesat_sub[:,2], icesat_sub[:,-1], xname='{} Elevation (m, WGS84)'.format(icesat_str_disp), yname='{} Elevation (m, WGS84)'.format(dataset_str),
                               units='m', lognorm=True,
                               savefile=os.path.join(figurepath,'{}_Density_{}_vs_ICESat.png'.format(continent, dataset_str)))

            # Height Differences vs. Elevation
            density(icesat_sub[:,2], diff, xname='{} Elevation (m, WGS84)'.format(icesat_str_disp), yname='{} - {} (m)'.format(dataset_str, icesat_str_disp),
                               ylim=diff_bounds_hist, units='m', lognorm=False, simline=False, fitline=False, showstats=False,
                               savefile=os.path.join(figurepath,'{}_Density_Diff_vs_ICESat.png'.format(continent)))
            
            if forested_flag:
                continent_id_all = np.append(continent_id_all, np.ones(len(diff))*i)
                diff_all = np.append(diff_all, diff)
                ground_diff_all = np.append(ground_diff_all, ground_diff)
                rh50_all = np.append(rh50_all, icesat_sub[:,3])
                rh75_all = np.append(rh75_all, icesat_sub[:,4])
                rh100_all = np.append(rh100_all, icesat_sub[:,5])
                trt_all = np.append(trt_all, icesat_sub[:,6])
            
            # Calculate tile-level statistics.
            tile_clat_file = os.path.join(outpath, continent+'_'+os.path.splitext(os.path.basename(icesat_file))[0]+'_tile_clat.npy')
            tile_clon_file = os.path.join(outpath, continent+'_'+os.path.splitext(os.path.basename(icesat_file))[0]+'_tile_clon.npy')
            tile_stats_file = os.path.join(outpath, continent+'_'+os.path.splitext(os.path.basename(icesat_file))[0]+'_tile_stats.npy')

            if os.path.isfile(tile_clat_file) and os.path.isfile(tile_clon_file) and os.path.isfile(tile_stats_file) and not overwrite_icesat_tile_stats:
                tile_clat = np.load(tile_clat_file)
                tile_clon = np.load(tile_clon_file)
                tile_stats = np.load(tile_stats_file)
            else:
                if forested_flag:
                    tile_clat, tile_clon, tile_stats = get_tile_level_stats(icesat_sub[:,0], icesat_sub[:,1], diff, ground_diff=ground_diff, canopyhgt=icesat_sub[:,5], trt=icesat_sub[:,6])
                else:
                    tile_clat, tile_clon, tile_stats = get_tile_level_stats(icesat_sub[:,0], icesat_sub[:,1], diff)
                np.save(tile_clat_file, tile_clat)
                np.save(tile_clon_file, tile_clon)
                np.save(tile_stats_file, tile_stats)

            # Append data for this continent to global arrays.
            if tile_clat_global is None:
                tile_clat_global = tile_clat.copy()
                tile_clon_global = tile_clon.copy()
                tile_stats_global = tile_stats.copy()
            else:
                tile_clat_global = np.append(tile_clat_global, tile_clat)
                tile_clon_global = np.append(tile_clon_global, tile_clon)
                tile_stats_global = np.append(tile_stats_global, tile_stats, axis=0)


            # Histograms of tile-level statistics.
            ind_valid = tile_stats[:,3] > min_tile_samples
            tile_clat_filt = tile_clat[ind_valid]
            tile_clon_filt = tile_clon[ind_valid]
            tile_stats_filt = tile_stats[ind_valid, :]

            hist(tile_stats_filt[:,0], xlim=bias_bounds_hist, xname='Bias of {} Tiles vs. {} (m)'.format(dataset_str, icesat_str_disp), numbins=40, units='m', plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_ICESat_Tiles_Bias.png'.format(continent, dataset_str)))

            hist(tile_stats_filt[:,2], xlim=rmse_bounds_hist, xname='RMSE of {} Tiles vs. {} (m)'.format(dataset_str, icesat_str_disp), numbins=40, units='m', plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_ICESat_Tiles_RMSE.png'.format(continent, dataset_str)))

            ind_valid = tile_stats[:,3] > 10
            tile_clat_filt = tile_clat[ind_valid]
            tile_clon_filt = tile_clon[ind_valid]
            tile_stats_filt = tile_stats[ind_valid, :]
            hist(np.log10(tile_stats_filt[:,3]), xlim=(1, 5), xname='log$_{10}$'+' of Number of ICESat Samples Within Each {} Tile (m)'.format(dataset_str), numbins=40, plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_ICESat_Tiles_Num.png'.format(continent, dataset_str)))
            plt.close('all')


            # Basemap of Raster-ICESat Shots
            if continent != 'Islands':
                if continent == 'NorthAmerica':
                    continent_display = 'North America'
                elif continent == 'SouthAmerica':
                    continent_display = 'South America'
                else:
                    continent_display = continent

                # Basemaps showing tile-level statistics vs. ICESat.
                saveroot_basemap_tile_stats = os.path.join(figurepath,'{}_Basemap_{}_minus_ICESat_Tiles'.format(continent, dataset_str))
                patch_size = 50000 # radius of circle used to plot each tile

                ind_valid = tile_stats[:,3] > min_tile_samples
                tile_clat_filt = tile_clat[ind_valid]
                tile_clon_filt = tile_clon[ind_valid]
                tile_stats_filt = tile_stats[ind_valid, :]

                plot_basemap(continent, tile_clat_filt, tile_clon_filt, tile_stats_filt[:,0], vmin=bias_bounds_map[0], vmax=bias_bounds_map[1],
                                   cmap='RdYlBu_r', cbar_label='Bias (m)', patch_size=patch_size,
                                   title='Bias of {} Tiles vs. {} \n for {} Areas of {}'.format(dataset_str, icesat_str_disp, forested_str, continent_display),
                                   ds_factor=1, savefile=saveroot_basemap_tile_stats+'_Bias.png')
                plt.close('all')

                plot_basemap(continent, tile_clat_filt, tile_clon_filt, tile_stats_filt[:,2], vmin=rmse_bounds_map[0], vmax=rmse_bounds_map[1],
                                   cmap='viridis', cbar_label='RMSE (m)', patch_size=patch_size,
                                   title='RMSE of {} Tiles vs. {} \n for {} Areas of {}'.format(dataset_str, icesat_str_disp, forested_str, continent_display),
                                   ds_factor=1, savefile=saveroot_basemap_tile_stats+'_RMSE.png')
                plt.close('all')

                if forested_flag:
                    plot_basemap(continent, tile_clat_filt, tile_clon_filt, tile_stats_filt[:,10], vmin=0, vmax=15,
                                       cmap='viridis', cbar_label='Vegetation Bias (m)', patch_size=patch_size,
                                       title='RMSE of {} Tiles vs. {} \n for {} Areas of {}'.format('NASADEM', icesat_str_disp, forested_str, continent_display),
                                       ds_factor=1, savefile=saveroot_basemap_tile_stats+'_Vegetation_Bias.png')
                    plt.close('all')

                    plot_basemap(continent, tile_clat_filt, tile_clon_filt, tile_stats_filt[:,6], vmin=0, vmax=30,
                                       cmap='viridis', cbar_label='Mean ICESat RH100 (m)', patch_size=patch_size,
                                       title='RMSE of {} Tiles vs. {} \n for {} Areas of {}'.format(dataset_str, icesat_str_disp, forested_str, continent_display),
                                       ds_factor=1, savefile=saveroot_basemap_tile_stats+'_MeanRH100.png')
                    plt.close('all')

                ind_valid = tile_stats[:,3] > 10
                tile_clat_filt = tile_clat[ind_valid]
                tile_clon_filt = tile_clon[ind_valid]
                tile_stats_filt = tile_stats[ind_valid, :]
                plot_basemap(continent, tile_clat_filt, tile_clon_filt, np.log10(tile_stats_filt[:,3]), vmin=1, vmax=5,
                                   cmap='viridis', cbar_label=r'log$_{10}$ of Number of ICESat Samples', patch_size=patch_size,
                                   title='Number of ICESat Samples Within Each {} Tile \n for {} Areas of {}'.format(dataset_str, forested_str, continent_display),
                                   ds_factor=1, savefile=saveroot_basemap_tile_stats+'_Num.png')
                plt.close('all')


        if plot_global:
            print('Plotting {} vs. ICESat Global Stats...'.format(dataset_str, continent))
            # Global Maps
            continent = 'Global'


            # For bias and RMSE, only show tiles with > min number of samples.
            ind_valid = tile_stats_global[:,3] > min_tile_samples
            tile_clat_global_filt = tile_clat_global[ind_valid]
            tile_clon_global_filt = tile_clon_global[ind_valid]
            tile_stats_global_filt = tile_stats_global[ind_valid, :]

            global_tile_clat_file = os.path.join(outpath, continent+'_'+os.path.splitext(os.path.basename(icesat_file))[0]+'_tile_clat_filt.npy')
            global_tile_clon_file = os.path.join(outpath, continent+'_'+os.path.splitext(os.path.basename(icesat_file))[0]+'_tile_clon_filt.npy')
            global_tile_stats_file = os.path.join(outpath, continent+'_'+os.path.splitext(os.path.basename(icesat_file))[0]+'_tile_stats_filt.npy')
            np.save(global_tile_clat_file, tile_clat_global_filt)
            np.save(global_tile_clon_file, tile_clon_global_filt)
            np.save(global_tile_stats_file, tile_stats_global_filt)

            # For number of samples within each tile, show more wider range of values.
            ind_valid = tile_stats_global[:,3] > 10
            tile_clat_global_all = tile_clat_global[ind_valid]
            tile_clon_global_all = tile_clon_global[ind_valid]
            tile_stats_global_all = tile_stats_global[ind_valid, :]

            global_tile_clat_file = os.path.join(outpath, continent+'_'+os.path.splitext(os.path.basename(icesat_file))[0]+'_tile_clat_all.npy')
            global_tile_clon_file = os.path.join(outpath, continent+'_'+os.path.splitext(os.path.basename(icesat_file))[0]+'_tile_clon_all.npy')
            global_tile_stats_file = os.path.join(outpath, continent+'_'+os.path.splitext(os.path.basename(icesat_file))[0]+'_tile_stats_all.npy')
            np.save(global_tile_clat_file, tile_clat_global_all)
            np.save(global_tile_clon_file, tile_clon_global_all)
            np.save(global_tile_stats_file, tile_stats_global_all)


            hist(tile_stats_global_filt[:,0], xlim=bias_bounds_hist, xname='Bias of {} Tiles vs. {} (m)'.format(dataset_str, icesat_str_disp), numbins=40, units='m', plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_ICESat_Tiles_Bias.png'.format(continent, dataset_str)))

            hist(tile_stats_global_filt[:,2], xlim=rmse_bounds_hist, xname='RMSE of {} Tiles vs. {} (m)'.format(dataset_str, icesat_str_disp), numbins=40, units='m', plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_ICESat_Tiles_RMSE.png'.format(continent, dataset_str)))

            hist(np.log10(tile_stats_global_all[:,3]), xlim=(1, 5), xname='log$_{10}$'+' of Number of ICESat Samples Within {} Tile'.format(dataset_str), numbins=40, plotmean=False,
                            showrms=False, savefile=os.path.join(figurepath,'{}_Hist_{}_minus_ICESat_Tiles_Num.png'.format(continent, dataset_str)))
            plt.close('all')

            # Overall Statistics (Percentiles of Absolute Bias and RMSE)
            print('Mean Tile Bias: {:.2f}'.format(np.nanmean(tile_stats_global_filt[:,0])))
            print('Mean MAE: {:.2f}'.format(np.nanmean(tile_stats_global_filt[:,4])))
            print('Mean RMSE: {:.2f}'.format(np.nanmean(tile_stats_global_filt[:,2])))

            print('10th Percentile of Global {} vs. ICESat Bias: {:.2f}'.format(dataset_str, np.nanpercentile((tile_stats_global_filt[:,0]), 10)))
            print('90th Percentile of Global {} vs. ICESat Bias: {:.2f}'.format(dataset_str, np.nanpercentile((tile_stats_global_filt[:,0]), 90)))

            print('50th Percentile of Global {} vs. ICESat MAE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,4], 50)))
            print('80th Percentile of Global {} vs. ICESat MAE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,4], 80)))
            print('90th Percentile of Global {} vs. ICESat MAE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,4], 90)))
            print('99th Percentile of Global {} vs. ICESat MAE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,4], 99)))

            print('50th Percentile of Global {} vs. ICESat RMSE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,2], 50)))
            print('80th Percentile of Global {} vs. ICESat RMSE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,2], 80)))
            print('90th Percentile of Global {} vs. ICESat RMSE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,2], 90)))
            print('99th Percentile of Global {} vs. ICESat RMSE: {:.2f}'.format(dataset_str, np.nanpercentile(tile_stats_global_filt[:,2], 99)))

            # Save geocoded stats rasters and shapefile.
            geotiff_basename = os.path.join(figurepath,'Global_{}_v_ICESat1_Tile_Stats'.format(dataset_str))
            create_geotiff_stats(tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt, geotiff_basename)

            output_shpfile = os.path.join(figurepath,'Global_{}_v_ICESat1_Tile_Stats.shp'.format(dataset_str))
            create_shp_stats(tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt, output_shpfile)

            # Basemaps showing tile-level statistics vs. ICESat.
            saveroot_basemap_tile_stats = os.path.join(figurepath,'{}_Basemap_{}_minus_ICESat_Tiles'.format(continent, dataset_str))

            plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,0], vmin=bias_bounds_map[0], vmax=bias_bounds_map[1],
                               cmap='RdYlBu_r', cbar_label='Bias (m)', patch_size=patch_size,
                               title='Mean Bias of {} Tiles vs. {} \n for {} Areas'.format(dataset_str, icesat_str_disp, forested_str),
                               ds_factor=1, savefile=saveroot_basemap_tile_stats+'_Bias.png')
            plt.close('all')


            plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,2], vmin=rmse_bounds_map[0], vmax=rmse_bounds_map[1],
                               cmap='viridis', cbar_label='RMSE (m)', patch_size=patch_size,
                               title='RMSE of {} Tiles vs. {}  \n for {} Areas'.format(dataset_str, icesat_str_disp, forested_str),
                               ds_factor=1, savefile=saveroot_basemap_tile_stats+'_RMSE.png')
            plt.close('all')

            plot_basemap(continent, tile_clat_global_all, tile_clon_global_all, np.log10(tile_stats_global_all[:,3]), vmin=1, vmax=5,
                               cmap='viridis', cbar_label=r'log$_{10}$ of Number of ICESat Samples', patch_size=patch_size,
                               title='Number of ICESat Samples Within Each {} Tile \n for {} Areas'.format(dataset_str, forested_str),
                               ds_factor=1, savefile=saveroot_basemap_tile_stats+'_Num.png')
            plt.close('all')

            # Vegetation Plots for Global Analysis
            if forested_flag:
                # Save files so we can compare vegetation bias between NASADEM compared to other datasets.
                h5_veg_output_file = os.path.join(outpath, 'vegetation_analysis_output.h5')
                with h5py.File(h5_veg_output_file, 'w') as hf_veg:
                    hf_veg.create_dataset('continent_id', data=continent_id_all)
                    hf_veg.create_dataset('diff', data=diff_all)
                    hf_veg.create_dataset('ground_diff', data=ground_diff_all)
                    hf_veg.create_dataset('rh50', data=rh50_all)
                    hf_veg.create_dataset('rh75', data=rh75_all)
                    hf_veg.create_dataset('rh100', data=rh100_all)
                    hf_veg.create_dataset('trt', data=trt_all)

                plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,10], vmin=0, vmax=15,
                                   cmap='viridis', cbar_label='Mean {} Vegetation Bias (m)'.format(dataset_disp), patch_size=patch_size,
                                   title='Mean Bias of {} Tiles vs. {} \n for {} Areas'.format('NASADEM', 'ICESat Ground', forested_str),
                                   ds_factor=1, savefile=saveroot_basemap_tile_stats+'_GroundBias.png')
                plt.close('all')

                # Basemaps showing mean ICESat RH100 and Trt. for each tile's vegetated areas.
                plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,6], vmin=0, vmax=30,
                                   cmap='viridis', cbar_label='Mean ICESat RH100 (m)', patch_size=patch_size,
                                   title='Mean ICESat RH100 of {} Tiles \n for {} Areas'.format(dataset_str, forested_str),
                                   ds_factor=1, savefile=saveroot_basemap_tile_stats+'_MeanRH100.png')
                plt.close('all')

                plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,7], vmin=0, vmax=30,
                                   cmap='viridis', cbar_label='Median ICESat RH100 (m)', patch_size=patch_size,
                                   title='Median ICESat RH100 of {} Tiles \n for {} Areas'.format(dataset_str, forested_str),
                                   ds_factor=1, savefile=saveroot_basemap_tile_stats+'_MedianRH100.png')
                plt.close('all')

                plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,8], vmin=0, vmax=75,
                                   cmap='viridis', cbar_label='Mean ICESat Ground Return Energy (%)', patch_size=patch_size,
                                   title='Mean ICESat Trt. of {} Tiles \n for {} Areas'.format(dataset_str, forested_str),
                                   ds_factor=1, savefile=saveroot_basemap_tile_stats+'_MeanTrt.png')
                plt.close('all')

                plot_basemap(continent, tile_clat_global_filt, tile_clon_global_filt, tile_stats_global_filt[:,9], vmin=0, vmax=75,
                                   cmap='viridis', cbar_label='Median ICESat Ground Return Energy (%)', patch_size=patch_size,
                                   title='Median ICESat Trt. of {} Tiles \n for {} Areas'.format(dataset_str, forested_str),
                                   ds_factor=1, savefile=saveroot_basemap_tile_stats+'_MedianTrt.png')
                plt.close('all')

                rh100_bin_edges = np.linspace(0, 60, num=13)
                rh100_bin_centers = (rh100_bin_edges[0:-1] + rh100_bin_edges[1:]) / 2

                ind_keep = np.ones_like(ground_diff_all, dtype='bool')
                for lower_bin, upper_bin in zip(rh100_bin_edges[0:-1], rh100_bin_edges[1:]):
                    ind_bin = (rh100_all >= lower_bin) & (rh100_all <= upper_bin)

                    if np.any(ind_bin):
                        ground_diff_temp = ground_diff_all[ind_bin]
                        th_lower, th_upper = np.nanpercentile(ground_diff_temp, [0.1, 99.9])

                        # Clip outlier thresholds to plot axes, so that we never discard data that would have been on the y-axis.
                        if th_upper < 60:
                            th_upper = 60

                        if th_lower > -10:
                            th_lower = -10

                        ind_outlier = ind_bin & (~np.isfinite(ground_diff_all) | (ground_diff_all < th_lower) | (ground_diff_all > th_upper))
                        ind_keep[ind_outlier] = False
                
                if np.any(ind_keep):
                    continent_id_all = continent_id_all[ind_keep]
                    diff_all = diff_all[ind_keep]
                    ground_diff_all = ground_diff_all[ind_keep]
                    rh50_all = rh50_all[ind_keep]
                    rh75_all = rh75_all[ind_keep]
                    rh100_all = rh100_all[ind_keep]
                    trt_all = trt_all[ind_keep]


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


                ind_fit = (rh100_all > 3.0)

                rmse_rh50 = np.sqrt(np.nanmean(np.square(ground_diff_all[ind_fit] - rh50_all[ind_fit])))
                rmse_rh75 = np.sqrt(np.nanmean(np.square(ground_diff_all[ind_fit] - rh75_all[ind_fit])))
                rmse_rh100 = np.sqrt(np.nanmean(np.square(ground_diff_all[ind_fit] - rh100_all[ind_fit])))

                print('Stats for Paper ({} vs. ICESat):'.format(dataset_str))
                print('')
                print('RH100:')
                fit_slope, fit_intercept, fit_r2, zifit_slope, zifit_r2 = do_fits_for_paper(rh100_all[ind_fit], ground_diff_all[ind_fit])
                print('Linear Fit. Slope: {}, Intercept: {}, r2: {}'.format(fit_slope, fit_intercept, fit_r2))
                print('Linear Fit w/ Zero Intercept.  Slope: {}, r2: {}'.format(zifit_slope, zifit_r2))
                print('RMSE vs. RH100: {}'.format(rmse_rh100))
                print('')
                print('RH75:')
                fit_slope, fit_intercept, fit_r2, zifit_slope, zifit_r2 = do_fits_for_paper(rh75_all[ind_fit], ground_diff_all[ind_fit])
                print('Linear Fit. Slope: {}, Intercept: {}, r2: {}'.format(fit_slope, fit_intercept, fit_r2))
                print('Linear Fit w/ Zero Intercept.  Slope: {}, r2: {}'.format(zifit_slope, zifit_r2))
                print('RMSE vs. RH75: {}'.format(rmse_rh75))
                print('')
                print('RH50')
                fit_slope, fit_intercept, fit_r2, zifit_slope, zifit_r2 = do_fits_for_paper(rh50_all[ind_fit], ground_diff_all[ind_fit])
                print('Linear Fit. Slope: {}, Intercept: {}, r2: {}'.format(fit_slope, fit_intercept, fit_r2))
                print('Linear Fit w/ Zero Intercept.  Slope: {}, r2: {}'.format(zifit_slope, zifit_r2))
                print('RMSE vs. RH50: {}'.format(rmse_rh50))
                print('')

                # Plot of Height Differences vs. RH100
                density(rh100_all, diff_all, xname='ICESat RH100 CHM (m)', yname='{} - {} (m)'.format(dataset_str, icesat_str_disp),
                                   units='m', lognorm=True, xlim=[0, 60], ylim=[-25, 25], fitline=True, showstats=False, simline=False,
                                   savefile=os.path.join(figurepath,'Global_Density_{}_{}_Diff_vs_ICESat_RH100.png'.format(dataset_str, icesat_str_disp)))

                density(rh100_all, ground_diff_all, xname='ICESat RH100 (m)', yname='{} - {} (m)'.format(dataset_disp, 'ICESat Ground'),
                                   units='m', lognorm=True, xlim=[0, 60], ylim=[-10, 60], fitline=True, showstats=False, simline=True, cov_regression=False,
                                   savefile=os.path.join(figurepath,'Global_Density_{}_GroundDiff_vs_ICESat_RH100.png'.format(dataset_str)))

                density(rh100_all, ground_diff_all, xname='ICESat RH100 (m)', yname='{} - {} (m)'.format(dataset_disp, 'ICESat Ground'),
                                   units='m', lognorm=True, xlim=[0, 60], ylim=[-10, 60], fitline=True, showstats=False, simline=True, cov_regression=False,
                                   #plot_binplot=True,
                                   savefile=os.path.join(figurepath,'Global_Density_{}_GroundDiff_vs_ICESat_RH100.png'.format(dataset_str)))


                multibinplot(rh100_all, diff_all, continent_id_all, continents, xname='ICESat RH100 (m)', yname='{} - {} (m)'.format(dataset_str, icesat_str_disp),
                                   xlim=[0, 60], ylim=[-5, 15], xbins=50, mincount=100,
                                   savefile=os.path.join(figurepath,'Global_Binplot_{}_{}_Diff_vs_ICESat_RH100.png'.format(dataset_str, icesat_str_disp)))

                multibinplot(rh100_all, ground_diff_all, continent_id_all, continents, xname='ICESat RH100 (m)', yname='{} - {} (m)'.format(dataset_str, 'ICESat Ground'),
                                   xlim=[0, 60], ylim=[-15, 60], xbins=50, mincount=100, simline=True,
                                   savefile=os.path.join(figurepath,'Global_Binplot_{}_GroundDiff_vs_ICESat_RH100.png'.format(dataset_str)))

                plt.close('all')

                # Plot of Height Difference vs. RH75
                density(rh75_all, diff_all, xname='ICESat RH75 (m)', yname='{} - {} (m)'.format(dataset_str, icesat_str_disp),
                                   units='m', lognorm=True, xlim=[0, 50], ylim=[-25, 25], fitline=True, showstats=False, simline=False,
                                   savefile=os.path.join(figurepath,'Global_Density_{}_{}_Diff_vs_ICESat_RH75.png'.format(dataset_str, icesat_str_disp)))

                density(rh75_all, ground_diff_all, xname='ICESat RH75 (m)', yname='{} - {} (m)'.format(dataset_str, 'ICESat Ground'),
                                   units='m', lognorm=True, xlim=[0, 60], ylim=[-10, 60], fitline=True, showstats=False, simline=True,
                                   savefile=os.path.join(figurepath,'Global_Density_{}_GroundDiff_vs_ICESat_RH75.png'.format(dataset_str)))

                multibinplot(rh75_all, diff_all, continent_id_all, continents, xname='ICESat RH75 (m)', yname='{} - {} (m)'.format(dataset_str, icesat_str_disp),
                                   xlim=[0, 60], ylim=[-15, 60], xbins=50, mincount=100,
                                   savefile=os.path.join(figurepath,'Global_Binplot_{}_{}_Diff_vs_ICESat_RH75.png'.format(dataset_str, icesat_str_disp)))

                multibinplot(rh75_all, ground_diff_all, continent_id_all, continents, xname='ICESat RH75 (m)', yname='{} - {} (m)'.format(dataset_str, 'ICESat Ground'),
                                   xlim=[0, 60], ylim=[-15, 60], xbins=50, mincount=100, simline=True,
                                   savefile=os.path.join(figurepath,'Global_Binplot_{}_GroundDiff_vs_ICESat_RH75.png'.format(dataset_str)))

                plt.close('all')

                # Plot of Height Difference vs. RH50
                density(rh50_all, diff_all, xname='ICESat RH50 CHM (m)', yname='{} - {} (m)'.format(dataset_str, icesat_str_disp),
                                   units='m', lognorm=True, xlim=[0, 50], ylim=[-25, 25], fitline=True, showstats=False, simline=False,
                                   savefile=os.path.join(figurepath,'Global_Density_{}_{}_Diff_vs_ICESat_RH50.png'.format(dataset_str, icesat_str_disp)))

                density(rh50_all, ground_diff_all, xname='ICESat RH50 CHM (m)', yname='{} - {} (m)'.format(dataset_str, 'ICESat Ground'),
                                   units='m', lognorm=True, xlim=[0, 60], ylim=[-10, 60], fitline=True, showstats=False, simline=True,
                                   savefile=os.path.join(figurepath,'Global_Density_{}_GroundDiff_vs_ICESat_RH50.png'.format(dataset_str)))

                multibinplot(rh50_all, diff_all, continent_id_all, continents, xname='ICESat RH50 (m)', yname='{} - {} (m)'.format(dataset_str, icesat_str_disp),
                                   xlim=[0, 45], ylim=[-5, 15], xbins=30, mincount=100,
                                   savefile=os.path.join(figurepath,'Global_Binplot_{}_{}_Diff_vs_ICESat_RH50.png'.format(dataset_str, icesat_str_disp)))

                multibinplot(rh50_all, ground_diff_all, continent_id_all, continents, xname='ICESat RH50 (m)', yname='{} - {} (m)'.format(dataset_str, 'ICESat Ground'),
                                   xlim=[0, 60], ylim=[-15, 60], xbins=30, mincount=100, simline=True,
                                   savefile=os.path.join(figurepath,'Global_Binplot_{}_GroundDiff_vs_ICESat_RH50.png'.format(dataset_str)))

                plt.close('all')