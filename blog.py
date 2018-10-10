"""
Script for computing values used in the blog writeup for this project
"""

from somerville_slope import * # bad practice, but convenient here

# number of lidar tiles
lidar_files = glob(os.path.join(LIDAR_DIR, '*'))
print(f'Number of lidar tiles: {len(lidar_files)}')


# total number of points in input dataset
fns = glob(os.path.join(LIDAR_DIR, '*.laz'))
counts = []
for fn in fns:
    res = subprocess.run(
        f'pdal info --metadata {fn}'.split(),
        stdout=subprocess.PIPE)
    counts.append(json.loads(res.stdout)['metadata']['count'])
num_lidar_pts = sum(counts) / 10**6
print(f'Total lidar points: {num_lidar_pts} M')

# total number of points in the analysis
# tree, junk = lidar_kdtree()
num_lidar_grnd_pts = tree.data.shape[0] / 10**6
print(f'Total lidar ground points: {num_lidar_pts} M')
print(f'Fraction ground points: {num_lidar_grnd_pts/num_lidar_pts}')

# average point density for one tile
res = subprocess.run(
    f'pdal info --metadata {fns[0]}'.split(),
    stdout=subprocess.PIPE)
tile_count = json.loads(res.stdout)['metadata']['count']
lidar_index = geopandas.read_file(INDEX_SHP).to_crs({"init": OUTPUT_CRS})
tile_area = lidar_index[lidar_index['Name'] == os.path.basename(fns[0])]['geometry'].tolist()[0].area
print(f'Point density: {tile_count/tile_area} pt/m2, {tile_area/tile_count} m2/pt')


