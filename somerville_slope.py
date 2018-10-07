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
import rasterio
import pickle
from scipy.spatial import cKDTree


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
OUTPUT_SOMERVILLE_KDTREE = 'data/output/somerville_kdtree.pkl'
OUTPUT_SOMERVILLE_ELEV_PREFIX = 'data/output/somerville_elev_'


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
                    # Note from LiDAR metadata: ... Default (Class 1), Ground (Class 2), Noise
                    # (Class 7), Water (Class 9), Ignored Ground (Class 10), Overlap Default
                    # (Class 17) and Overlap Ground (Class 18).
                    {
                      "type":"filters.range",
                      "limits":"Classification[2:2], Classification[9:10], Classification[18:18]"
                    },
                    {
                        "type": "writers.gdal",
                        "resolution": 1,
                        "radius": 2,
                        "output_type": "max",
                        "filename": output_file,
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

        # reformat as x,y,z array of ground points
        print(f'{input_file}: Get x,y,z array')
        pts = np.column_stack((pts['X'], pts['Y'], pts['Z']))
        
        # save data as numpy file
        print(f'{input_file}: Save as {output_file}')
        np.save(output_file, pts)


def lidar_kdtree(load=True):
    """
    Create / load KDTree containing all (filtered) LiDAR data

    Arguments:
        load: bool, set True to attempt to load previous results from pickle files

    Return: tree, tree
        tree: scipy.spatial.cKDTree object for x,y in pts
        zpts: Nx1 numpy array, z coordinates for all points in study area, note that x,y are stored in tree.data
    """
    if load and os.path.exists(OUTPUT_SOMERVILLE_KDTREE):
        # load data from pickle
        print(f'Reading stored results from: {OUTPUT_SOMERVILLE_KDTREE}')
        with open(OUTPUT_SOMERVILLE_KDTREE, 'rb') as fp:
            data = pickle.load(fp)
            tree = data['tree']
            zpts = data['zpts']

    else: 
        # generate pts data from tile inputs
        pt_arrays = []
        for npy_file in glob(os.path.join(OUTPUT_LIDAR_DIR, '*.npy')):
            print(f'Read: {npy_file}')
            pt_arrays.append(np.load(npy_file))
        pts = np.row_stack(pt_arrays)
        del pt_arrays
        # build KDTree for x,y points
        print(f'Building KDTree for {pts.shape[0]/10**6}M points')
        tree = cKDTree(pts[:,:2])
        zpts = pts[:,2]
        # save as pickle
        print(f"Saving as pickle: {OUTPUT_SOMERVILLE_KDTREE}")
        with open(OUTPUT_SOMERVILLE_KDTREE, 'wb') as fp:
            pickle.dump({'zpts': zpts, 'tree': tree}, fp)

    return tree, zpts


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
    """
    Read Somerville mask raster from file
    
    Returns: mask, x_vec, y_vec, meta
        mask: 2D numpy array, Somerville footprint 
        x_vec, y_vec: 1D numpy arrays, coordinates for mask array
        meta: dict, metadata from the mask raster, useful for writing related rasters
    """
    with rasterio.open(OUTPUT_SOMERVILLE_MASK_GTIF) as src:
        # read mask data
        mask = src.read(indexes=1)

        # generate coordinate vectors (and check them against original)
        bnds = src.bounds # outer bounds, points are 1/2 pixel in from this edge
        y_vec = np.arange(bnds.bottom + 0.5*OUTPUT_RES_Y,
                          bnds.top    - 0.5*OUTPUT_RES_Y + OUTPUT_RES_Y,
                          OUTPUT_RES_Y)
        y_vec = y_vec[::-1] # reverse direction to match image coords
        x_vec = np.arange(bnds.left  + 0.5*OUTPUT_RES_X,
                          bnds.right - 0.5*OUTPUT_RES_X + OUTPUT_RES_X,
                          OUTPUT_RES_X)
        assert src.xy(row=150, col=150) == (x_vec[150], y_vec[150])

        # fetch raster metadata (useful for writing related rasters)
        meta = src.meta.copy()

    return mask, x_vec, y_vec, meta


def create_somerville_elevation_geotiff(knn=16, agg='median'):
    """Compute gridded elev using NN-median filter as save geotiff"""
    # get points and KDTree
    tree, zpts = lidar_kdtree(load=True)

    # prepare output grid from somerville mask raster
    mask, x_vec, y_vec, meta = read_somerville_mask_geotiff()
    mask = mask.astype(np.bool)
    elev = np.zeros(mask.shape, dtype=np.float32)
    elev[:] = np.nan

    # find kNN for all grid points
    x_grd, y_grd = np.meshgrid(x_vec, y_vec, indexing='xy')
    xy_mask = np.column_stack((x_grd[mask], y_grd[mask]))
    nn_dist, nn_idx = tree.query(xy_mask, k=knn) # returns indexes into original data

    if agg == 'median':
        # compute elevation as local medians
        elev[mask] = np.median(zpts[nn_idx], axis=1)
        output_file = f'{OUTPUT_SOMERVILLE_ELEV_PREFIX}_median_{knn}.gtif'
    elif agg == 'mean':
        elev[mask] = np.mean(zpts[nn_idx], axis=1)
        output_file = f'{OUTPUT_SOMERVILLE_ELEV_PREFIX}_mean_{knn}.gtif'
    else:
        raise ValueError(f'Invalid choice for input arg "agg" = {agg}')

    # write results to geotiff
    meta.update({
        'driver': 'GTiff',
        'dtype': 'float32',
        })
    with rasterio.open(output_file, 'w', **meta) as elev_raster:
        elev_raster.write(elev, 1)


