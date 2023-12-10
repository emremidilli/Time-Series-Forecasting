#!/bin/bash

patch_size=24
pool_size_trend=2

main() {
    model_id=$1
    cd ../app_input_pipeline/
    echo "starting to build input pipeline for " $model_id

    docker-compose run --rm app_input_pipeline \
        pre_train.py \
        --model_id=$model_id \
        --patch_size=$patch_size \
        --pool_size_trend=$pool_size_trend

    echo "Bulding input pipelines is successfull for " $model_id
}

main "model_01"
