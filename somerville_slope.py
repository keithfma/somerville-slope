"""
Somerville MA LiDAR Slope Analysis

This (importable) module contains all subroutines used
"""
import os
import geopandas
import wget
from glob import glob
import json
import pdal
import numpy as np
import math
import subprocess


# constants
MA_TOWNS_SHP = 'data/massgis_towns/data/TOWNS_POLY.shp'
INDEX_SHP = 'data/noaa_lidar_index/data/2013_2014_usgs_post_sandy_ma_nh_ri_index.shp'
LIDAR_DIR = 'data/noaa_lidar/dist'
LIDAR_CRS = 'EPSG:4152' # NAD83(HARN), see: https://coast.noaa.gov/htdata/lidar1_z/geoid12b/data/4800/

OUTPUT_CRS = 'EPSG:32619' # UTM 19N coord ref sys, good for eastern MA
OUTPUT_RES_X = 1 # meters
OUTPUT_RES_Y = 1 # meters
OUTPUT_LIDAR_DIR = 'data/output/lidar'
OUTPUT_SOMERVILLE_SHP = 'data/output/somerville_boundary.shp'
OUTPUT_INDEX_SOMERVILLE_SHP = 'data/output/somerville_lidar_index.shp'
OUTPUT_SOMERVILLE_MASK_GTIF = 'data/output/somerville_mask.gtif'


def lidar_download():
    """Find and download LiDAR tiles for Somerville MA"""

    # load MA towns and project to standard
    ma_towns = geopandas.read_file(MA_TOWNS_SHP).to_crs({"init": OUTPUT_CRS})
    somerville = ma_towns.set_index('TOWN').loc['SOMERVILLE']

    # load NOAA post-sandy LiDAR tile footprints, clip to somerville
    lidar_tiles = geopandas.read_file(INDEX_SHP).to_crs({'init': OUTPUT_CRS})
    in_somerville = lidar_tiles.geometry.intersects(somerville.geometry)
    lidar_tiles_somerville = lidar_tiles[in_somerville]

    # write out Somerville LiDAR tiles as shapefile
    lidar_tiles_somerville.to_file(OUTPUT_INDEX_SOMERVILLE_SHP)

    # download Somerville tiles
    urls = lidar_tiles_somerville['URL'].values.tolist()
    for ii, url in enumerate(urls):
        print(f'\nDownloading tile {ii+1} / {len(urls)}')
        wget.download(url=url.strip(), out=LIDAR_DIR)


def lidar_preprocess():
    """Read and preprocess (new) LiDAR tiles"""
    for input_file in glob(os.path.join(LIDAR_DIR, '*.laz')):
        
        # compute output name
        output_file = os.path.join(
            OUTPUT_LIDAR_DIR,
            os.path.splitext(os.path.basename(input_file))[0] + '.npy'
            )
       
        # check for existing output and skip if found
        if os.path.isfile(output_file):
            print(f'{input_file}: output exists, skipping')
            continue
        
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
                        "out_srs": OUTPUT_CRS,
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
        
        # save data as numpy file
        print(f'{input_file}: Save as {output_file}')
        np.save(output_file, pts)


# def lidar_kdtree(load=True):
#     """Create / load KDTree containing all (filtered) LiDAR data"""
#     # TODO: add logic for loading existing KDTree
#     pass
# 
# # read and concatenate all pre-processed lidar tiles
# pt_arrays = []
# for npy_file in glob(os.path.join(OUTPUT_LIDAR_DIR, '*.npy')):
#     pt_arrays.append(np.load(npy_file))
# pts = np.row_stack(pt_arrays)
# del pt_arrays


def create_output_array():
    """
    Generate empty array and coordinates for Somerville bounding box

    Returns: empty_grid, x_vec, y_vec, raster_origin, pixel_width, pixel_height
        empty_grid:
        x_vec, y_vec:
        raster_origin:
        pixel_width, pixel_height:
    """

    # load somerville geometry and get coord data
    somerville = geopandas.read_file(OUTPUT_SOMERVILLE_SHP)
    somerville_poly = somerville['geometry'][0]
    x_min, y_min, x_max, y_max = somerville_poly.bounds

    # generate coord info required to write geotiff
    raster_origin = (math.floor(x_min), math.floor(y_min))
    pixel_width = OUTPUT_RES_X
    pixel_height = OUTPUT_RES_Y

    # generate coordinate vectors
    x_vec = np.arange(math.floor(x_min), math.ceil(x_max) + 1, 1.0)
    y_vec = np.arange(math.floor(y_min), math.ceil(y_max) + 1, 1.0)

    # generate empty array to be populated
    empty_grid = np.zeros((y_vec.shape[0], x_vec.shape[0]))
    empty_grid[:] = np.nan
    
    # done!
    return empty_grid, x_vec, y_vec, raster_origin, pixel_width, pixel_height


def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array):
    """
    Create GeoTiff from numpy array 
    
    Modified from https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#create-raster-from-array

    Arguments: TODO
    
    Returns: TODO
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32) # hardcoded to single precision
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(OUTPUT_CRS) # hardcoded to standard coord sys
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def create_somerville_shp():
    """Generate shapefile with Somerville geometry"""
    ma_towns = geopandas.read_file(MA_TOWNS_SHP).to_crs({"init": OUTPUT_CRS})
    ma_towns[ma_towns['TOWN'] == 'SOMERVILLE'].to_file(OUTPUT_SOMERVILLE_SHP)


def create_somerville_mask_geotiff():
    """Create a mask raster for Somerville footprint"""
    # load somerville geometry and get coord data
    somer = geopandas.read_file(OUTPUT_SOMERVILLE_SHP)
    somer_poly = somer['geometry'][0]

    # rasterize using command-line tool
    cmd = ['gdal_rasterize', 
        '-burn',  1,
        '-of', 'GTiff',
        '-a_nodata', 0,
        '-te', *somer_poly.bounds,
        '-tr', OUTPUT_RES_X, OUTPUT_RES_Y,
        '-ot', 'Byte',
        OUTPUT_SOMERVILLE_SHP,
        OUTPUT_SOMERVILLE_MASK_GTIF,
        ]
    subprocess.run([str(arg) for arg in cmd], check=True)


def read_somerville_mask_geotiff():
    """Return empty array, coordinate vectors, and rasterio metadata"""
    pass



# build 2D KDTree from point x, y

# TODO: how to keep track of z?

# # find NN for all grid points
# tree = cKDTree(xy) 
# xy_grd = np.hstack([x_grd.reshape((-1,1)), y_grd.reshape((-1,1))])
# nn_dist, nn_idx = tree.query(xy_grd, k=16)
#
# # compute local medians
# z_grds.append(np.median(zz[nn_idx], axis=1).reshape(x_grd.shape))

