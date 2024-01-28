#!/bin/bash

pool_size_trend=24
sigma=3.0
scale_data="N"
begin_scalar=-1.0
end_scalar=-1.0

main() {
    dataset_id=$1
    cd ../app_input_pipeline/
    echo "starting to build input pipeline for " $dataset_id

    docker-compose run --rm app_input_pipeline \
        fine_tune.py \
        --dataset_id=$dataset_id \
        --pool_size_trend=$pool_size_trend \
        --sigma=$sigma \
        --scale_data=$scale_data \
        --begin_scalar=$begin_scalar \
        --end_scalar=$end_scalar \

    echo "Bulding input pipelines is successfull for " $dataset_id
}

main "dataset_03"