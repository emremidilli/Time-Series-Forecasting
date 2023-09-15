#!/bin/bash

cd ../app_raw_data_formatter/

CHANNELS=("EURUSD" "GBPUSD" "USDCAD")

for channel in ${CHANNELS[@]}; do
    echo "starting to process " $channel

    docker-compose run --rm app_data_formatter \
        02_build_datasets.py \
        --channel=$channel \
        --target_group=ticker \
        --forecast_horizon=120 \
        --lookback_coefficient=4 \
        --step_size=30 \
        --test_size=500

    echo "02_build_datasets is successfull for "$channel

    docker-compose run --rm app_data_formatter \
        03_convert_time_indices_to_date_features.py \
        --channel=$channel \
        --raw_frequency=m \
        --datetime_features="['month', 'day', 'dayofweek', 'hour', 'minute']"

    echo "03_convert_time_indices_to_date_features is successfull for "$channel
done

echo "done"
