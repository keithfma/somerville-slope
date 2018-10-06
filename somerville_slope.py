"""
Somerville MA LiDAR Slope Analysis

This (importable) module contains all subroutines used
"""
import os
import geopandas
import wget
from glob import glob
import json
import subprocess
import pdal
import numpy as np


# constants
MA_TOWNS_SHP = 'data/massgis_towns/data/TOWNS_POLY.shp'
INDEX_SHP = 'data/noaa_lidar_index/data/2013_2014_usgs_post_sandy_ma_nh_ri_index.shp'
INDEX_SOMERVILLE_SHP = 'maps/noaa_lidar_index_somerville.shp'
LIDAR_BASE_DIR = 'data/noaa_lidar'
LIDAR_DIST_DIR = f'{LIDAR_BASE_DIR}/dist'
LIDAR_DATA_DIR = f'{LIDAR_BASE_DIR}/data'
LIDAR_CRS = 'EPSG:4152' # NAD83(HARN), see: https://coast.noaa.gov/htdata/lidar1_z/geoid12b/data/4800/
STD_CRS = 'EPSG:32619' # UTM 19N coord ref sys, good for eastern MA


def lidar_download():
    """Find and download LiDAR tiles for Somerville MA"""

    # load MA towns and project to standard
    ma_towns = geopandas.read_file(MA_TOWNS_SHP).to_crs(STD_CRS)
    somerville = ma_towns.set_index('TOWN').loc['SOMERVILLE']

    # load NOAA post-sandy LiDAR tile footprints, clip to somerville
    lidar_tiles = geopandas.read_file(INDEX_SHP).to_crs({'init': STD_CRS})
    in_somerville = lidar_tiles.geometry.intersects(somerville.geometry)
    lidar_tiles_somerville = lidar_tiles[in_somerville]

    # write out Somerville LiDAR tiles as shapefile
    lidar_tiles_somerville.to_file(INDEX_SOMERVILLE_SHP)

    # download Somerville tiles
    urls = lidar_tiles_somerville['URL'].values.tolist()
    for ii, url in enumerate(urls):
        print(f'\nDownloading tile {ii+1} / {len(urls)}')
        wget.download(url=url.strip(), out=LIDAR_DIST_DIR)


def lidar_preprocess():
    """Read and preprocess LiDAR tiles"""
    for input_file in glob(os.path.join(LIDAR_DIST_DIR, '*.laz')):
        # read and preprocss -> numpy array
        print(f'\n{input_file}: Read and preprocess data')
        pipeline_json = json.dumps(
            {
                "pipeline": [
                    {
                        "type": "readers.las",
                        "filename": input_file,
                    }, {
                        "type": "filters.reprojection",
                        "in_srs": LIDAR_CRS,
                        "out_srs": STD_CRS,
                    },
                    {
                        "type": "filters.outlier",
                        "method": "statistical",
                        "mean_k": 12,
                        "multiplier": 3,
                    },
                ]
            }
        )
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.validate()
        pipeline.execute()
        num_arrays = len(pipeline.arrays)
        assert num_arrays == 1, f"Unexpected length for pipeline.arrays = {num_arrays}"
        pts = pipeline.arrays[0]

        # find ground points
        # # Note from LiDAR metadata: ... Default (Class 1), Ground (Class 2), Noise
        # # (Class 7), Water (Class 9), Ignored Ground (Class 10), Overlap Default
        # # (Class 17) and Overlap Ground (Class 18).
        print(f'{input_file}Find ground points from')
        last_or_only = pts['ReturnNumber'] == pts['NumberOfReturns']  
        default_or_ground_or_water = np.logical_or(
            pts['Classification'] == 1,
            pts['Classification'] == 2,
            pts['Classification'] == 9)
        mask = np.logical_and(last_or_only, default_or_ground_or_water)

        # reformat as x,y,z array of ground points
        print(f'{input_file}: Get x,y,z array')
        pts = np.column_stack((pts['X'][mask], pts['Y'][mask], pts['Z'][mask]))
        
        # compute output name
        output_file = os.path.join(
            LIDAR_DATA_DIR,
            os.path.splitext(os.path.basename(input_file))[0] + '.npy'
            )
        
        # save data as numpy file
        print(f'{input_file}: Save as {output_file}')
        np.save(output_file, pts)


def lidar_kdtree(load=True):
    """Create / load KDTree containing all (filtered) LiDAR data"""
    # TODO: add logic for loading existing KDTree
    pass

# read and concatenate all pre-processed lidar tiles
pt_arrays = []
for npy_file in glob(os.path.join(LIDAR_DATA_DIR, '*.npy')):
    pt_arrays.append(np.load(npy_file))
pts = np.row_stack(pt_arrays)
del pt_arrays


# build 2D KDTree from point x, y

# TODO: how to keep track of z?

# # find NN for all grid points
# tree = cKDTree(xy) 
# xy_grd = np.hstack([x_grd.reshape((-1,1)), y_grd.reshape((-1,1))])
# nn_dist, nn_idx = tree.query(xy_grd, k=16)
#
# # compute local medians
# z_grds.append(np.median(zz[nn_idx], axis=1).reshape(x_grd.shape))

