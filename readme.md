# nasadem_val

**This is not actively maintained**; it was tested with python 3.7.

These are functions to assist in the filtering and appending of global data to the lidar shot data from ICESat-2 and GEDI. Tested with python 3.7.

The basic workflow divides items into dem 1 degree tiles. From there, we use the GeoDataFrame objects to:

1. Extract raster data from the pixels covered by the lidar shot
2. Filter data based on auxiliary raster data, dem slope, and the lidar shot metadata. 


## Installation
Clone this repository. Install the environment using `mamba` i.e. `mamba env update -f environment.yml`. Then install the library with `pip install .`

## Usage

See the jupyter notebooks in `notebooks/`. These will assume data in a private server.