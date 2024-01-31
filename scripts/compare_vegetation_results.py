#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparing Vegetation Results from NASADEM, SRTM V3, and GLO-30

Script to compare vegetation results for various DEMs vs. ICESat-1, ICESat-2, and
GEDI.

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
from osgeo import gdal, osr, ogr
import matplotlib.pyplot as plt
import h5py
import kapok.plot

# Matplotlib Style Options
plt.style.use('seaborn-white')
plt.rcParams['lines.markeredgewidth'] = 3
plt.rcParams['lines.markersize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams.update({'font.size': 18})



# hdf5 files -- UPDATE TO YOUR FILE LOCATION
root_path = '/Volumes/Disk/validation_results/'

# paths containing vegetation hdf5 files created by other scripts:
h5_paths = np.array(['NASADEM_srtm_validation_vegshots_v9_2023_10_18',
                     'SRTMV3_validation_vegshots_v9_2023_10_18',
                     'COP_validation_vegshots_v9_2023_10_18',
                     'NASADEM_srtm_validation_icesat2_vegshots_v9_2023_10_19',
                     'SRTMV3_validation_icesat2_vegshots_v9_2023_10_19',
                     'COP_validation_icesat2_vegshots_v9_2023_10_19',
                     'NASADEM_srtm_validation_gedi_vegshots_v4_2023_10_19',
                     'SRTMV3_validation_gedi_vegshots_v4_2023_10_19',
                     'COP_validation_gedi_vegshots_v4_2023_10_19'])

num_bins = 7
rh_bin_edges = np.linspace(0, 60, num=num_bins)
rh_bin_edges = np.append(rh_bin_edges, np.inf)
rh_bin_centers = (rh_bin_edges[0:-1] + rh_bin_edges[1:])/2
rh_bin_centers[-1] = rh_bin_centers[-2] + (rh_bin_centers[-2] - rh_bin_centers[-3])
rh_bin_labels = np.array(['<10', '10-20', '20-30', '30-40', '40-50', '50-60', '>60']) # display names for each size category

bias_mean = np.zeros((len(h5_paths), len(rh_bin_centers)), dtype='float32')
bias_stddev = np.zeros((len(h5_paths), len(rh_bin_centers)), dtype='float32')
bias_mean_rh50 = np.zeros((len(h5_paths), len(rh_bin_centers)), dtype='float32')
bias_stddev_rh50 = np.zeros((len(h5_paths), len(rh_bin_centers)), dtype='float32')

for path_num, path in enumerate(h5_paths):
    h5_file = os.path.join(root_path, path + '/vegetation_analysis_output.h5')
    with h5py.File(h5_file, 'r') as hdf:

        rh50 = hdf['rh50'][:]

        if 'gedi' in h5_file:
            canopy_height = hdf['rh98'][:]
        else:
            canopy_height = hdf['rh100'][:]

        rh50_ratio = rh50/canopy_height
        rh50_ratio = np.clip(rh50_ratio, 1e-2, 1)

        ground_diff = hdf['ground_diff'][:]

        for bin_num in range(bias_mean.shape[1]):
            ind_bin = (canopy_height >= rh_bin_edges[bin_num]) & (canopy_height <= rh_bin_edges[bin_num+1])
            bias_mean[path_num, bin_num] = np.nanmean(ground_diff[ind_bin])
            bias_stddev[path_num, bin_num] = np.nanstd(ground_diff[ind_bin])

            ind_rh50_bin = (rh50 >= rh_bin_edges[bin_num]) & (rh50 <= rh_bin_edges[bin_num+1])
            bias_mean_rh50[path_num, bin_num] = np.nanmean(ground_diff[ind_rh50_bin])
            bias_stddev_rh50[path_num, bin_num] = np.nanstd(ground_diff[ind_rh50_bin])




outpath = os.path.join(root_path, 'results')
if not os.path.isdir(outpath):
    os.makedirs(outpath)


bin_adj = 2.0 # amount to shift markers so they're not all on top of each other


# Plot vs. ICESat RH100
plt.figure()
plt.errorbar(rh_bin_centers-bin_adj, bias_mean[0], yerr=bias_stddev[0], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='NASADEM')
plt.errorbar(rh_bin_centers, bias_mean[1], yerr=bias_stddev[1], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='SRTM V3')
plt.errorbar(rh_bin_centers+bin_adj, bias_mean[2], yerr=bias_stddev[2], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='Copernicus')
plt.ylabel('DEM - ICESat Ground (m)')
plt.xlabel('ICESat RH100 (m)')
plt.legend(loc='upper left', frameon=True, fontsize=10)
plt.ylim([0, 70])
plt.xlim([0, 70])
plt.xticks(rh_bin_centers)
plt.gca().set_xticklabels(rh_bin_labels)
plt.savefig(os.path.join(outpath,'binplot_icesat_rh100_vegetation_bias.png'), dpi=125, bbox_inches='tight')


# Plot vs. ICESat-2 RH100
plt.figure()
plt.errorbar(rh_bin_centers-bin_adj, bias_mean[3], yerr=bias_stddev[3], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='NASADEM')
plt.errorbar(rh_bin_centers, bias_mean[4], yerr=bias_stddev[4], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='SRTM V3')
plt.errorbar(rh_bin_centers+bin_adj, bias_mean[5], yerr=bias_stddev[5], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='Copernicus')
plt.ylabel('DEM - ICESat-2 Ground (m)')
plt.xlabel('ICESat-2 RH100 (m)')
plt.legend(loc='upper left', frameon=True, fontsize=10)
plt.ylim([0, 70])
plt.xlim([0, 70])
plt.xticks(rh_bin_centers)
plt.gca().set_xticklabels(rh_bin_labels)
plt.savefig(os.path.join(outpath,'binplot_icesat2_rh100_vegetation_bias.png'), dpi=125, bbox_inches='tight')


# Plot vs. GEDI RH100
plt.figure()
plt.errorbar(rh_bin_centers-bin_adj, bias_mean[6], yerr=bias_stddev[6], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='NASADEM')
plt.errorbar(rh_bin_centers, bias_mean[7], yerr=bias_stddev[7], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='SRTM V3')
plt.errorbar(rh_bin_centers+bin_adj, bias_mean[8], yerr=bias_stddev[8], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='Copernicus')
plt.ylabel('DEM - GEDI Ground (m)')
plt.xlabel('GEDI RH98 (m)')
plt.legend(loc='upper left', frameon=True, fontsize=10)
plt.ylim([0, 70])
plt.xlim([0, 70])
plt.xticks(rh_bin_centers)
plt.gca().set_xticklabels(rh_bin_labels)
plt.savefig(os.path.join(outpath,'binplot_gedi_rh98_vegetation_bias.png'), dpi=125, bbox_inches='tight')


# Plot vs. GEDI RH50
plt.figure()
plt.errorbar(rh_bin_centers-bin_adj, bias_mean_rh50[6], yerr=bias_stddev_rh50[6], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='NASADEM')
plt.errorbar(rh_bin_centers, bias_mean_rh50[7], yerr=bias_stddev_rh50[7], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='SRTM V3')
plt.errorbar(rh_bin_centers+bin_adj, bias_mean_rh50[8], yerr=bias_stddev_rh50[8], fmt='.', elinewidth=2, capsize=2, capthick=2, barsabove=False, label='Copernicus')
plt.ylabel('DEM - GEDI Ground (m)')
plt.xlabel('GEDI RH50 (m)')
plt.legend(loc='upper left', frameon=True, fontsize=10)
plt.ylim([0, 70])
plt.xlim([0, 70])
plt.xticks(rh_bin_centers)
plt.gca().set_xticklabels(rh_bin_labels)
plt.savefig(os.path.join(outpath,'binplot_gedi_rh50_vegetation_bias.png'), dpi=125, bbox_inches='tight')
