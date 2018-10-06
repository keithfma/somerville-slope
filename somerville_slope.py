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

# SCRATCH --------------------------------------------------------------------

#    pipeline_json = json.dumps({
#        "pipeline": [
#            {
#                "type": "readers.las",
#                "filename": laz_file,
#            }, {
#                "type": "filters.reprojection",
#                "out_srs": f"EPSG:{cfg.PRJ_SRID}",
#            }, {
#                "type": "filters.chipper",
#                "capacity": cfg.LIDAR_CHIP,
#            }, {
#                "type": "writers.pgpointcloud",
#                "connection": f"host={cfg.PSQL_HOST} dbname={cfg.LIDAR_DB} user={cfg.PSQL_USER} password={cfg.PSQL_PASS} port={cfg.PSQL_PORT}",
#                "table": cfg.LIDAR_TABLE,
#                "compression": "dimensional",
#                "srid": cfg.PRJ_SRID,
#                "output_dims": "X,Y,Z,ReturnNumber,NumberOfReturns,Classification", # reduce data volume
#                "scale_x": 0.01, # precision in meters
#                "scale_y": 0.01,
#                "scale_z": 0.01, 
#                "offset_x": 0, # TODO: select a smarter value
#                "offset_y": 0,
#                "offset_z": 0,
#            }
#        ]
#    })
#    subprocess.run(['pdal', 'pipeline', '--stdin'], input=pipeline_json.encode('utf-8'))

#     # Note from LiDAR metadata: ... Default (Class 1), Ground (Class 2), Noise
#     # (Class 7), Water (Class 9), Ignored Ground (Class 10), Overlap Default
#     # (Class 17) and Overlap Ground (Class 18).
# 
#     # build output grid spanning bbox
#     x_vec = np.arange(math.floor(x_min), math.floor(x_max), cfg.SURFACE_RES_M)   
#     y_vec = np.arange(math.floor(y_min), math.floor(y_max), cfg.SURFACE_RES_M)   
#     x_grd, y_grd = np.meshgrid(x_vec, y_vec)
# 
#     # retrieve data, including a pad on all sides
#     pts = lidar.retrieve(x_min-PAD, x_max+PAD, y_min-PAD, y_max+PAD)
# 
#     # extract ground points
#     grnd_idx = []
#     for idx, pt in enumerate(pts):
#         if pt[3] == pt[4] and pt[5] in {1, 2, 9}:
#             # last or only return, classified as "default", "ground" or "water"
#             grnd_idx.append(idx)
#     grnd_pts = pts[grnd_idx, :3]
#     
#     # extract upper surface points
#     surf_idx = []
#     for idx, pt in enumerate(pts):
#         if (pt[3] == 1 or pt[4] == 1) and pt[5] in {1, 2, 9}:
#             # first or only return, classified as "default", "ground", or "water" 
#             surf_idx.append(idx)
#     surf_pts = pts[surf_idx, :3]
#     del pts
# 
#     z_grds = []
#     for pts in [grnd_pts, surf_pts]: 
#         # extract [x, y] and z arrays
#         xy = pts[:, :2]
#         zz = pts[:,  2]
# 
#         # find NN for all grid points
#         tree = cKDTree(xy) 
#         xy_grd = np.hstack([x_grd.reshape((-1,1)), y_grd.reshape((-1,1))])
#         nn_dist, nn_idx = tree.query(xy_grd, k=16)
# 
#         # compute local medians
#         z_grds.append(np.median(zz[nn_idx], axis=1).reshape(x_grd.shape))

#     # build pipeline definition and execute
#     filename = uuid.uuid4().hex
#     pipeline_json= json.dumps({
#         "pipeline":[
#             {
#                 "type": "readers.pgpointcloud",
#                 "connection": f"host={cfg.PSQL_HOST} dbname={cfg.LIDAR_DB} user={cfg.PSQL_USER} password={cfg.PSQL_PASS} port={cfg.PSQL_PORT}",
#                 "table": cfg.LIDAR_TABLE,
#                 "column": "pa",
#                 # TODO: Holy shit! I think that I had the args wrong all along here! Confirm change.
#                 # "where": f"PC_Intersects(pa, ST_MakeEnvelope({xmin}, {xmax}, {ymin}, {ymax}, {cfg.PRJ_SRID}))",
#                 "where": f"PC_Intersects(pa, ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {cfg.PRJ_SRID}))",
#             }, {
#                 "type": "writers.text",
#                 "format": "csv",
#                 "filename": filename,
#             }
#           ]
#         })
#     subprocess.run(['pdal', 'pipeline', '--stdin'], input=pipeline_json.encode('utf-8'))
#     
#     # read resulting file to numpy, then delete it
#     array = np.loadtxt(filename, delimiter=',', dtype=float, skiprows=1)
#     os.remove(filename)

