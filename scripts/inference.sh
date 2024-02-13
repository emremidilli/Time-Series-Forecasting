#!/bin/bash

main() {
    dataset_id=$1
    pre_processor_id=$2
    model_id=$3

    lb_dir="./tsf-bin/02_formatted_data/$dataset_id/lb_test.npy"
    ts_dir="./tsf-bin/02_formatted_data/$dataset_id/ts_test.npy"

    pre_processor_dir="./tsf-bin/03_preprocessing/$pre_processor_id/input_preprocessor/"
    input_save_dir="./tsf-bin/05_inference/$dataset_id/input/"

    echo "inference is started"
    cd ../app_input_pipeline/

    docker-compose run --rm app_input_pipeline \
        inference.py \
        --lb_dir=$lb_dir \
        --ts_dir=$ts_dir \
        --pre_processor_dir=$pre_processor_dir \
        --save_dir=$input_save_dir

    output_save_dir="./tsf-bin/05_inference/$dataset_id/output/"
    cd ../app_training/
    docker-compose run --rm app_training \
        inference.py \
        --model_id=$model_id \
        --input_dir=$input_save_dir \
        --output_dir=$output_save_dir

    echo "inference is completed."
}

main "ds_universal_ETTh1_96_4_S" "ds_universal_ETTh1_96_4_S_ft" "model_20240203_06_ft"
