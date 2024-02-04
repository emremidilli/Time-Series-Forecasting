#!/bin/bash

cd ../app_data_formatter/

test_size=0.20

main() {
    model_id=$1
    dataset_id=$2
    list_of_covariates=$3
    forecast_horizon=$4
    lookback_coefficient=$5
    step_size=$6
    raw_frequency=$7
    datetime_features=$8
    echo "starting to process " $model_id

    docker-compose run --rm app_data_formatter \
        build_datasets.py \
        --model_id=$model_id \
        --dataset_id=$dataset_id \
        --list_of_covariates=$list_of_covariates \
        --forecast_horizon=$forecast_horizon \
        --lookback_coefficient=$lookback_coefficient \
        --step_size=$step_size \
        --test_size=$test_size \
        --raw_frequency=$raw_frequency \
        --datetime_features=$datetime_features

    echo "successfull for "$model_id
}

main "ds_20240203_04" "ETTh1" "['OT']" 96 4 1 "h" "['month','day','dayofweek']"
