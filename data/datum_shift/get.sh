#!/bin/bash

mkdir -p dist
mkdir -p data
cd dist

declare -a urls=("http://download.osgeo.org/proj/proj-datumgrid-1.8.zip"
                "http://download.osgeo.org/proj/proj-datumgrid-europe-1.1.zip"
                "http://download.osgeo.org/proj/proj-datumgrid-north-america-1.1.zip"
                "http://download.osgeo.org/proj/proj-datumgrid-oceania-1.0.zip"
                )

for url in "${urls[@]}"; do
    filename=$(basename $url)
    echo "$url -> $filename"
    wget $url
    unzip $filename -d ../data
done

