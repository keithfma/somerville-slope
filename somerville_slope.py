"""
Somerville MA LiDAR Slope Analysis

This (importable) module contains all subroutines used
"""

# TODO: too many spurious results -- try a morphological filter on the gridded
#   elev / slope results to handle ugly edges of house footprints -- try
#   increasing the number of points required for a fit

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
from warnings import warn
import shutil


# constants
MA_TOWNS_SHP = 'data/massgis_towns/data/TOWNS_POLY.shp'
SOMERVILLE_PARCELS_SHP = 'data/massgis_parcels/data/M274TaxPar.shp'
INDEX_SHP = 'data/noaa_lidar_index/data/2013_2014_usgs_post_sandy_ma_nh_ri_index.shp'
LIDAR_DIR = 'data/noaa_lidar/dist'
LIDAR_CRS = 'EPSG:4152' # NAD83(HARN), see: https://coast.noaa.gov/htdata/lidar1_z/geoid12b/data/4800/
FT_TO_M = 0.3048
NBR_RADIUS = 15*FT_TO_M 
FIT_MIN_PTS = 0.75*math.pi*NBR_RADIUS**2 # 30 ft radius around point at least 3/4 populated
SLOPE_PCT_THRESHOLD = 25
IMPACTED_THRESHOLD = 5 # m2 above threshold to label as impacted

OUTPUT_CRS = 'EPSG:32619' # UTM 19N coord ref sys, good for eastern MA
OUTPUT_RES_X = 1 # meters
OUTPUT_RES_Y = 1 # meters
OUTPUT_DIR = 'data/output'
OUTPUT_LIDAR_DIR = f'{OUTPUT_DIR}/lidar'
OUTPUT_SOMERVILLE_SHP = f'{OUTPUT_DIR}/somerville_boundary.shp'
OUTPUT_INDEX_SOMERVILLE_SHP = f'{OUTPUT_DIR}/somerville_lidar_index.shp'
OUTPUT_SOMERVILLE_MASK_GTIF = f'{OUTPUT_DIR}/somerville_mask.gtif'
OUTPUT_SOMERVILLE_KDTREE = f'{OUTPUT_DIR}/somerville_kdtree.pkl'
OUTPUT_SOMERVILLE_ELEV_PREFIX = f'{OUTPUT_DIR}/somerville_elev'
OUTPUT_SOMERVILLE_SLOPE_PCT_GTIF = f'{OUTPUT_DIR}/somerville_slope_pct_lstsq_30ft.gtif'
OUTPUT_SOMERVILLE_SLOPE_DIR_GTIF = f'{OUTPUT_DIR}/somerville_slope_dir_lstsq_30ft.gtif'
OUTPUT_SOMERVILLE_PARCELS_SHP = f'{OUTPUT_DIR}/someville_parcels.shp'
OUTPUT_SOMERVILLE_PARCELS_GTIF = f'{OUTPUT_DIR}/somerville_parcels.gtif'
OUTPUT_SOMERVILLE_PARCELS_LABELED_SHP = f'{OUTPUT_DIR}/somerville_parcels_labeled.shp'


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

    Return: tree, zpts
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


def read_geotiff(gtif):
    """
    Read raster from file

    Arguments:
        gtif: string, path to input file to be read
    
    Returns: mask, x_vec, y_vec, meta
        mask: 2D numpy array, Somerville footprint 
        x_vec, y_vec: 1D numpy arrays, coordinates for mask array
        meta: dict, metadata from the mask raster, useful for writing related rasters
    """
    with rasterio.open(gtif) as src:
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


def create_somerville_elev_slope_geotiffs():
    """Compute gridded elev and gradient grids using least-squares, save geotiffs"""

    # get points and KDTree
    tree, zpts = lidar_kdtree(load=True)

    # prepare output grids from somerville mask raster
    mask, x_vec, y_vec, meta = read_geotiff(OUTPUT_SOMERVILLE_MASK_GTIF)
    mask = mask.astype(np.bool)
    elev = np.zeros(mask.shape, dtype=np.float32)
    elev[:] = np.nan
    slope_dir = elev.copy()
    slope_pct = elev.copy()

    # populate all grid points
    nrows, ncols = elev.shape
    for ii in range(nrows):

        # progress monitor
        if ii % 100 == 0 or ii == nrows-1:
            print(f'Row {ii} / {nrows}')

        for jj in range(ncols):
            if mask[ii, jj]:

                # get point coords
                this_x = x_vec[jj]
                this_y = y_vec[ii]

                # get all pts within 15 ft (yields 30-foot diameter circle ROI)
                nbr_idx = tree.query_ball_point((this_x, this_y), NBR_RADIUS)
                nbr_num = len(nbr_idx)
                if nbr_num < FIT_MIN_PTS:
                    continue

                # find best-fit plane to points
                fit = np.linalg.lstsq(
                    a=np.column_stack(( np.ones((nbr_num, 1)), tree.data[nbr_idx] )),
                    b=zpts[nbr_idx],
                    rcond=None
                    )[0]

                # extract elevation (evaluate best fit plane at this point)
                elev[ii,jj] = fit[0] + fit[1]*this_x + fit[2]*this_y

                # extract slope magnitude (vector magnitude is m/m, times 100 to percent grade)
                # NOTE: confusingly, percent grade can be > 100, see: https://en.wikipedia.org/wiki/Grade_(slope)
                slope_pct[ii,jj] = np.sqrt(fit[1]*fit[1] + fit[2]*fit[2])*100
                
                # extract slope direction
                slope_dir[ii,jj] = np.degrees(np.arctan2(fit[2], fit[1]))

    # write results to geotiff
    meta.update({
        'driver': 'GTiff',
        'dtype': 'float32',
        })
    with rasterio.open(f'{OUTPUT_SOMERVILLE_ELEV_PREFIX}_lstsq_30ft.gtif', 'w', **meta) as elev_raster:
        elev_raster.write(elev, 1)
    with rasterio.open(OUTPUT_SOMERVILLE_SLOPE_PCT_GTIF, 'w', **meta) as slope_pct_raster:
        slope_pct_raster.write(slope_pct, 1)
    with rasterio.open(OUTPUT_SOMERVILLE_SLOPE_DIR_GTIF, 'w', **meta) as slope_dir_raster:
        slope_dir_raster.write(slope_dir, 1)



def create_somerville_parcel_shp():
    """Write shapefile containing select Somerville parcels"""
    # read original parcels shp
    parcels = geopandas.read_file(SOMERVILLE_PARCELS_SHP).to_crs({"init": OUTPUT_CRS})

    # drop select parcel types
    is_row = parcels['POLY_TYPE'] == 'ROW'
    is_rail_row = parcels['POLY_TYPE'] == 'RAIL_ROW'
    is_water = parcels['POLY_TYPE'] == 'WATER'
    rel_parcels = parcels.drop(parcels[is_row | is_rail_row | is_water].index)

    # add numeric ID as a feature
    rel_parcels.reset_index(drop=True)
    rel_parcels['ID'] = rel_parcels.index

    # write to shapefile
    rel_parcels.to_file(OUTPUT_SOMERVILLE_PARCELS_SHP)


def create_somerville_parcel_geotiff(): 
    """Create a raster of Somerville parcel IDs -- match other rasters format"""
    # load somerville geometry and get coord data
    somer = geopandas.read_file(OUTPUT_SOMERVILLE_SHP)
    somer_poly = somer['geometry'][0]

    # rasterize using command-line tool
    cmd = ['gdal_rasterize', 
        '-a', 'ID',
        '-of', 'GTiff',
        '-a_nodata', -9999,
        '-te', *somer_poly.bounds,
        '-tr', OUTPUT_RES_X, OUTPUT_RES_Y,
        '-ot', 'Int32',
        OUTPUT_SOMERVILLE_PARCELS_SHP,
        OUTPUT_SOMERVILLE_PARCELS_GTIF,
        ]
    subprocess.run([str(arg) for arg in cmd], check=True)


def create_labeled_parcel_shp():
    """Write Somerville parcel shapefile with slope counts and labels"""
    # load slope and parcel raster data
    slope_pct_array, x_vec, y_vec, meta = read_geotiff(OUTPUT_SOMERVILLE_SLOPE_PCT_GTIF)
    parcels_array, x_vec, y_vec, meta = read_geotiff(OUTPUT_SOMERVILLE_PARCELS_GTIF)

    # load somerville parcels shapefile
    parcels = geopandas.read_file(OUTPUT_SOMERVILLE_PARCELS_SHP)

    # count above-threshold slope pixels in each parcel
    parcels_count = {parcel_id: 0 for parcel_id in parcels['ID']}
    parcels_count[-9999] = 0 # special case for no-data pixels
    for ii in range(parcels_array.shape[0]):
        for jj in range(parcels_array.shape[1]):
            if slope_pct_array[ii, jj] >= SLOPE_PCT_THRESHOLD:
                parcels_count[parcels_array[ii, jj]] += 1
            else:
                parcels_count[parcels_array[ii, jj]] += 0
    del parcels_count[-9999] # special case for no-data pixels

    # convert counter to a pandas DataFrame and add classification
    counts_df = geopandas.pd.DataFrame.from_dict(parcels_count, orient='index').reset_index()
    counts_df.rename({'index': 'ID', 0: 'M2_ABOVE_25_PCT_SLOPE'}, axis='columns', inplace=True)
    counts_df['IMPACTED'] = counts_df['M2_ABOVE_25_PCT_SLOPE'] >= IMPACTED_THRESHOLD

    # join dataframes
    parcels.set_index('ID', drop=True, inplace=True)  
    counts_df.set_index('ID', drop=True, inplace=True)  
    parcels_combined = parcels.join(counts_df, how='left')

    # write results to shapefile
    parcels_combined['IMPACTED'] = parcels_combined['IMPACTED'].astype(int)
    parcels_combined.to_file(OUTPUT_SOMERVILLE_PARCELS_LABELED_SHP)


