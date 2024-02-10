#!/bin/bash

pool_size_trend=24
sigma=3.0
scale_data="N"

main() {
    input_dataset_id=$1
    output_dataset_id=$2
    cd ../app_input_pipeline/
    echo "starting to build input pipeline for " $input_dataset_id

    docker-compose run --rm app_input_pipeline \
        fine_tune.py \
        --input_dataset_id=$input_dataset_id \
        --output_dataset_id=$output_dataset_id \
        --pool_size_trend=$pool_size_trend \
        --sigma=$sigma \
        --scale_data=$scale_data

    echo "Pipeline is built with the name of " $output_dataset_id
}

main "ds_20240203_large" "ds_20240203_large_ft"

main "ds_20240203_few_shot" "ds_20240203_few_shot_ft"
