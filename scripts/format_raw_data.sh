#!/bin/bash

cd ../app_raw_data_formatter/

target_group="ticker"
step_size=1
test_size=0.20

main() {
    channel=$1
    forecast_horizon=$2
    lookback_coefficient=$3
    raw_frequency=$4
    datetime_features=$5
    echo "starting to process " $channel

    docker-compose run --rm app_data_formatter \
        02_build_datasets.py \
        --channel=$channel \
        --target_group=$target_group \
        --forecast_horizon=$forecast_horizon \
        --lookback_coefficient=$lookback_coefficient \
        --step_size=$step_size \
        --test_size=$test_size

    echo "02_build_datasets is successfull for "$channel

    docker-compose run --rm app_data_formatter \
        03_convert_time_indices_to_date_features.py \
        --channel=$channel \
        --raw_frequency=$raw_frequency \
        --datetime_features=$datetime_features

    echo "03_convert_time_indices_to_date_features is successfull for "$channel

    echo "done"
}

main "ETTh1" 168 2 "h" "['month','day','dayofweek','hour']"
# main "ETTh2" 168 2 "h" "['month','day','dayofweek','hour']"
# main "ETTm1" 168 2 "15m" "['month','day','dayofweek','hour','minute']"
# main "ETTm2" 168 2 "15m" "['month','day','dayofweek','hour','minute']"
# main "electricity" 168 2 "h" "['month','day','dayofweek','hour']"
# main "traffic" 168 2 "h" "['month','day','dayofweek','hour']"
# main "weather" 168 2 "10m" "['month','day','dayofweek','hour','minute']"