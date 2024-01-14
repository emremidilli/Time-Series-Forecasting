#!/bin/bash

pool_size_trend=24
sigma=3.0
scale_data="N"

main() {
    model_id=$1
    cd ../app_input_pipeline/
    echo "starting to build input pipeline for " $model_id

    docker-compose run --rm app_input_pipeline \
        pre_train.py \
        --model_id=$model_id \
        --pool_size_trend=$pool_size_trend \
        --sigma=$sigma \
        --scale_data=$scale_data

    echo "Bulding input pipelines is successfull for " $model_id
}

main "model_03"
