"""
Script for computing values used in the blog writeup for this project
"""

from somerville_slope import * # bad practice, but convenient here
from matplotlib import pyplot as plt
import subprocess
import os

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
tree, zz = lidar_kdtree()
xx = tree.data[:,0]
yy = tree.data[:,1]
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

# figure: decimated points for selected area
x_lim = [326100, 326300]
y_lim = [4695200, 4695300]
inside_lims = np.logical_and(
    np.logical_and(xx >= x_lim[0], xx <= x_lim[1]),
    np.logical_and(yy >= y_lim[0], yy <= y_lim[1]))
fig = plt.figure(
    figsize=(10, 6),
    dpi=120)
hs = plt.scatter(
    x=xx[inside_lims]-x_lim[0],
    y=yy[inside_lims]-y_lim[0],
    c=zz[inside_lims],
    marker='.',
    s=1)
plt.colorbar(
    mappable=hs,
    shrink=0.5,
    label='elevation [m]')
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim(0, x_lim[1]-x_lim[0])
ax.set_ylim(0, y_lim[1]-y_lim[0])
ax.set_xticks(range(0, 201, 100))
ax.set_yticks(range(0, 101, 100))
plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')
plt.savefig(
    fname="results/subset_elevation_points.png",
    bbox_inches='tight',
    dpi=200,
    format='png')

# figure: example of local planar fit
# # get points
pt_x = 326374 
pt_y = 4695300
nbr_idx = tree.query_ball_point(x=(pt_x, pt_y), r=15*FT_TO_M)
nbr_x = xx[nbr_idx]
nbr_y = yy[nbr_idx]
nbr_z = zz[nbr_idx]
# # perform fit
fit = np.linalg.lstsq(
    a=np.column_stack(( np.ones((len(nbr_idx), 1)), nbr_x, nbr_y)),
    b=nbr_z,
    rcond=None
    )[0]
fit_c = fit[0]
fit_a = fit[1]
fit_b = fit[2]
# # generate plane
plane_xx, plane_yy = np.meshgrid(
    np.arange(min(roi_x), max(roi_x)+0.25, 0.25),
    np.arange(min(roi_y), max(roi_y)+0.25, 0.25))
plane_zz = fit_a*plane_xx + fit_b*plane_yy + fit_c
# # generate ROI
roi_theta = np.arange(0, 2*np.pi, 0.01)
roi_r = 15*FT_TO_M*np.ones(roi_theta.shape)
roi_x = roi_r * np.cos(roi_theta) + pt_x
roi_y = roi_r * np.sin(roi_theta) + pt_y
roi_z = fit_a*roi_x + fit_b*roi_y + fit_c
# # create figure
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(
    figsize=(8, 6.25),
    dpi=100)
ax = fig.add_subplot(111, projection='3d')
x0 = min(roi_x)
y0 = min(roi_y)
# # plot plane
ax.plot_surface(
    plane_xx-x0, plane_yy-y0, plane_zz,
    color='C0',
    alpha=0.3)
# # plot roi
ax.plot(
    roi_x-x0, roi_y-y0, roi_z,
    color='C0',
    label='15-ft Radius') 
# # plot observations
nbr_fit_z = fit_a*nbr_x + fit_b*nbr_y + fit_c
nbr_resid = nbr_z - nbr_fit_z
is_above = nbr_resid >= 0
is_below = nbr_resid < 0
ax.scatter(
    nbr_x[is_above]-x0, nbr_y[is_above]-y0, nbr_z[is_above],
    marker='+',
    color='C1',
    s=30,
    label='LiDAR, Above Plane')
ax.scatter(
    nbr_x[is_below]-x0, nbr_y[is_below]-y0, nbr_z[is_below],
    marker='_',
    color='C1',
    s=30,
    label='LiDAR, Below Plane')
# # format appearance
plt.xlim(min(roi_x)-x0, max(roi_x)-x0)
plt.ylim(min(roi_y)-y0, max(roi_y)-y0)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.view_init(azim=33)
ax.legend()
plt.savefig(
    fname="results/delete_me.png",
    dpi=200,
    format='png')
# # trim whitespace (tight save does not work in 3D)
subprocess.run('convert results/delete_me.png -trim -bordercolor White -border 30x30 results/example_fit.png'.split(' '))
os.remove('results/delete_me.png')


