#!/bin/bash
# Run after downloading 9GB data file - and renaming to "cervical_cnn_training_dataset_raw" and moving into working directory
# Download Data: https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed/data?select=im_Superficial-Intermediate

if ![-d results]; then
  mkdir results
fi

mkdir cervical_cnn_training_dataset_clean
cp -r cervical_cnn_training_dataset_raw/im_Dyskeratotic/* cervical_cnn_training_dataset_clean
cp -r cervical_cnn_training_dataset_raw/im_Parabasal/* cervical_cnn_training_dataset_clean
cp -r cervical_cnn_training_dataset_raw/im_Koilocytotic/* cervical_cnn_training_dataset_clean
cp -r cervical_cnn_training_dataset_raw/im_Superficial-Intermediate/* cervical_cnn_training_dataset_clean
cp -r cervical_cnn_training_dataset_raw/im_Metaplastic/* cervical_cnn_training_dataset_clean
cd cervical_cnn_training_dataset_clean
find ./* -type d -name 'CROPPED' -prune -o -type f -exec rm -f {} +
find ./ -type f -name "*.dat" -delete

mv ./im_Dyskeratotic/CROPPED/* ./im_Dyskeratotic/
mv ./im_Parabasal/CROPPED/* ./im_Parabasal/
mv ./im_Koilocytotic/CROPPED/* ./im_Koilocytotic/
mv ./im_Superficial-Intermediate/CROPPED/* ./im_Superficial-Intermediate
mv ./im_Metaplastic/CROPPED/* ./im_Metaplastic/

find ./ -type d -name "CROPPED" -delete