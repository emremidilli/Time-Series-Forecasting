#!/bin/bash

channel=$1
patch_size=24
pool_size_reduction=2
pool_size_trend=2
nr_of_bins=10
pre_train_ratio=0.10
mask_scalar=0.001
begin_scalar=-1.0
end_scalar=-1.0

main() {
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
        --mask_scalar=$mask_scalar \
        --begin_scalar=$begin_scalar \
        --end_scalar=$end_scalar

    echo "Bulding input pipelines is successfull for " $channel
}

main
