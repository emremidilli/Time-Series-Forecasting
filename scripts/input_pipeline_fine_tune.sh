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

main "ds_universal_ETTh1_96_4_M" "ds_universal_ETTh1_96_4_M_ft"

main "ds_universal_ETTh2_96_4_M" "ds_universal_ETTh2_96_4_M_ft"

main "ds_universal_ETTm1_96_4_M" "ds_universal_ETTm1_96_4_M_ft"

main "ds_universal_ETTm2_96_4_M" "ds_universal_ETTm2_96_4_M_ft"