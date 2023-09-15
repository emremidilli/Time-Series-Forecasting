#!/bin/bash

channel=$1
patch_size=30
pool_size_reduction=5
pool_size_trend=2
nr_of_bins=8
pre_train_ratio=0.25
quantiles="[0.10,0.50,0.90]"
mask_scalar=0.53

cd ../app_input_pipeline/
echo "starting to build input pipeline for " $channel

docker-compose run --rm app_input_pipeline \
    pre_train.py \
    --channel=$channel \
    --patch_size=$patch_size \
    --pool_size_reduction=$pool_size_reduction \
    --pool_size_trend=$pool_size_trend \
    --nr_of_bins=$nr_of_bins \
    --pre_train_ratio=$pre_train_ratio

echo "Input pipeline for pre-traning is built."

docker-compose run --rm app_input_pipeline \
    fine_tune.py \
    --channel=$channel \
    --patch_size=$patch_size \
    --pool_size_reduction=$pool_size_reduction \
    --pool_size_trend=$pool_size_trend \
    --nr_of_bins=$nr_of_bins \
    --quantiles=$quantiles \
    --mask_scalar=$mask_scalar


echo "Bulding input pipelines is successfull for " $channel
