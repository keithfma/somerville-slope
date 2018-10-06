#!/bin/bash

mkdir -p dist
mkdir -p data
cd dist
wget http://download.massgis.digital.mass.gov/shapefiles/l3parcels/L3_SHP_M274_SOMERVILLE.zip
unzip L3_SHP_M274_SOMERVILLE.zip -d ../data
mv ../data/L3_SHP_M274_SOMERVILLE/* ../data
rmdir ../data/L3_SHP_M274_SOMERVILLE
