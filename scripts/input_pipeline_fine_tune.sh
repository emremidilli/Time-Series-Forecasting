#!/bin/bash

pool_size_trend=24
sigma=3.0
scale_data="N"
begin_scalar=-1.0
end_scalar=-1.0

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
        --scale_data=$scale_data \
        --begin_scalar=$begin_scalar \
        --end_scalar=$end_scalar \

    echo "Pipeline is built with the name of " $output_dataset_id
}

main "dataset_03" "dataset_04"