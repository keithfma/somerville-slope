"""
Run all analysis steps
"""

import somerville_slope as ss

# ss.lidar_download()
# ss.lidar_preprocess()
# ss.lidar_kdtree()
# ss.create_somerville_shp()
# ss.create_somerville_parcel_shp():
# ss.create_somerville_mask_geotiff()
ss.create_somerville_elev_slope_geotiffs()
# ss.create_somerville_parcel_geotiff()
ss.create_labeled_parcel_shp()

