#!/bin/bash

model_id=$1
patch_size=24
pool_size_reduction=2
pool_size_trend=2
mask_scalar=0.001
begin_scalar=-1.0
end_scalar=-1.0

main() {
    cd ../app_input_pipeline/
    echo "starting to build input pipeline for " $model_id

    docker-compose run --rm app_input_pipeline \
        fine_tune.py \
        --model_id=$model_id \
        --patch_size=$patch_size \
        --pool_size_reduction=$pool_size_reduction \
        --pool_size_trend=$pool_size_trend \
        --mask_scalar=$mask_scalar \
        --begin_scalar=$begin_scalar \
        --end_scalar=$end_scalar

    echo "Bulding input pipelines is successfull for " $model_id
}

main
