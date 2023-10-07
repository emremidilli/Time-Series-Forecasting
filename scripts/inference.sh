#!/bin/bash

channel=$1

lb_dir="./tsf-bin/02_formatted_data/$channel/lb_test.npy"
ts_dir="./tsf-bin/02_formatted_data/$channel/ts_test.npy"
pre_processor_dir="./tsf-bin/03_preprocessing/$channel/fine_tune/input_preprocessor/"
input_save_dir="./tsf-bin/05_inference/$channel/input/"
model_dir="./tsf-bin/04_artifacts/$channel/fine_tune/saved_model/"
output_save_dir="./tsf-bin/05_inference/$channel/output/"
nr_of_forecasting_steps=168
begin_scalar=-1.0

cd ../app_input_pipeline/


main() {
    echo "inference is started"

    docker-compose run --rm app_input_pipeline \
        inference.py \
        --lb_dir=$lb_dir \
        --ts_dir=$ts_dir \
        --pre_processor_dir=$pre_processor_dir \
        --save_dir=$input_save_dir

    cd ../app_training/
    docker-compose run --rm app_training \
        inference.py \
        --input_dataset_dir=$input_save_dir \
        --model_dir=$model_dir \
        --output_save_dir=$output_save_dir \
        --nr_of_forecasting_steps=$nr_of_forecasting_steps \
        --begin_scalar=$begin_scalar

    echo "inference is completed."
}

main

