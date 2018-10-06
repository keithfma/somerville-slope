"""
Somerville MA LiDAR Slope Analysis

This (importable) module contains all subroutines used
"""
import os
import geopandas
import wget
import glob


# constants
MA_TOWNS_SHP_FILE = 'data/massgis_towns/data/TOWNS_POLY.shp'
LIDAR_INDEX_SHP_FILE = 'data/noaa_lidar_index/data/2013_2014_usgs_post_sandy_ma_nh_ri_index.shp'
LIDAR_SOMERVILLE_SHP_FILE = 'maps/noaa_lidar_index_somerville.shp'
LIDAR_SOMERVILLE_BASE_DIR = 'data/noaa_lidar'
LIDAR_SOMERVILLE_DIST_DIR = f'{LIDAR_SOMERVILLE_BASE_DIR}/dist'
STD_CRS = {'init': 'EPSG:32619'} # UTM 19N coord ref sys, good for eastern MA


def lidar_download():
    """Find and download LiDAR tiles for Somerville MA"""

    # load MA towns and project to standard
    ma_towns = geopandas.read_file(MA_TOWNS_SHP_FILE).to_crs(STD_CRS)
    somerville = ma_towns.set_index('TOWN').loc['SOMERVILLE']

    # load NOAA post-sandy LiDAR tile footprints, clip to somerville
    lidar_tiles = geopandas.read_file(LIDAR_INDEX_SHP_FILE).to_crs(STD_CRS)
    in_somerville = lidar_tiles.geometry.intersects(somerville.geometry)
    lidar_tiles_somerville = lidar_tiles[in_somerville]

    # write out Somerville LiDAR tiles as shapefile
    lidar_tiles_somerville.to_file(LIDAR_SOMERVILLE_SHP_FILE)

    # download Somerville tiles
    urls = lidar_tiles_somerville['URL'].values.tolist()
    for ii, url in enumerate(urls):
        print(f'Downloading tile {ii+1} / {len(urls)}')
        wget.download(url=url.strip(), out=LIDAR_SOMERVILLE_DIST_DIR)


def lidar_merge():
    """Clean and merge LiDAR tiles to single dataset"""
    # TODO: re-use parasol project pre-processing pipeline, write results to file
    pass


def lidar_kdtree():
    """Create / load KDTree containing all (filtered) LiDAR data"""
    # TODO: re-use code from parasol gridding routine
    pass
