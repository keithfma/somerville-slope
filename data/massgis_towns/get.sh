#!/bin/bash

mkdir -p dist
mkdir -p data
cd dist
wget http://download.massgis.digital.mass.gov/shapefiles/state/towns.zip
unzip towns.zip -d ../data
