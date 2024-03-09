#!/bin/bash

cd ../app_universal_data/

main() {
    input_dataset_id=$1
    output_dataset_id=$2
    forecast_horizon=$3
    lookback_coefficient=$4
    features=$5

    echo "starting to process " $model_id

    docker-compose run --rm app_universal_data \
        get_universal_dataset.py \
        --input_dataset_id=$input_dataset_id \
        --output_dataset_id=$output_dataset_id \
        --forecast_horizon=$forecast_horizon \
        --lookback_coefficient=$lookback_coefficient \
        --features=$features
}

main "etth1" "ds_universal_ETTh1_96_4_S" 96 4 "S"
main "etth1" "ds_universal_ETTh1_96_4_M" 96 4 "M"

main "etth2" "ds_universal_ETTh2_96_4_S" 96 4 "S"
main "etth2" "ds_universal_ETTh2_96_4_M" 96 4 "M"

main "ettm1" "ds_universal_ETTm1_96_4_S" 96 4 "S"
main "ettm1" "ds_universal_ETTm1_96_4_M" 96 4 "M"

main "ettm2" "ds_universal_ETTm2_96_4_S" 96 4 "S"
main "ettm2" "ds_universal_ETTm2_96_4_M" 96 4 "M"
