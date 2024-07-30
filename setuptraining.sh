#!/bin/bash
# only Run after downloading 2005 DTU/HERLEV DATABASE: https://mde-lab.aegean.gr/index.php/downloads/ and moving into working directory

find ./ -type f -name "*-d.bmp" -delete
mkdir results
mkdir logs
mkdir model_data
mkdir model_data/high_dys_or_carcinoma
mkdir model_data/normal_cells
cp carcinoma_in_situ/* model_data/high_dys_or_carcinoma
cp severe_dysplastic/* model_data/high_dys_or_carcinoma
cp normal_columnar/* model_data/normal_cells
cp normal_intermediate/* model_data/normal_cells
cp normal_superficiel/* model_data/normal_cells