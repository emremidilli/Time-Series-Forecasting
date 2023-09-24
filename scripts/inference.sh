#!/bin/bash

channel=$1

lb_dir="./tsf-bin/02_training_datasets/$channel/lb_test.npy"
ts_dir="./tsf-bin/02_training_datasets/$channel/ts_test.npy"
pre_processor_dir="./tsf-bin/03_preprocessing/$channel/fine_tune/input_preprocessor/"
input_save_dir="./tsf-bin/05_inference/$channel/input/"
model_dir="./tsf-bin/04_artifacts/$channel/fine_tune/saved_model/"
output_save_dir="./tsf-bin/05_inference/$channel/output/"

echo "inference is started"

cd ../app_input_pipeline/
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
    --output_save_dir=$output_save_dir

echo "inference is completed."