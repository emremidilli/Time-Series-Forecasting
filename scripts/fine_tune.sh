#!/bin/bash

model_id="model_20240203_05_ft"
pre_trained_model_id="model_20240203_05_pt_comp_tre_sea_cl"
dataset_id="ds_20240203_large_ft_scaled"
resume_training="N"
validation_rate=0.00
mini_batch_size=128
learning_rate=0.001
clip_norm=1.0
nr_of_epochs=100
fine_tune_backbone="N"

cd ../app_training/

main() {
    echo "starting to fine-tune " $model_id

    docker-compose run --rm app_training \
        fine_tune.py \
        --model_id=$model_id \
        --pre_trained_model_id=$pre_trained_model_id \
        --dataset_id=$dataset_id \
        --resume_training=$resume_training \
        --validation_rate=$validation_rate \
        --mini_batch_size=$mini_batch_size \
        --learning_rate=$learning_rate \
        --clip_norm=$clip_norm \
        --nr_of_epochs=$nr_of_epochs \
        --fine_tune_backbone=$fine_tune_backbone

    echo "fine-tuning is successfull for " $model_id
}

main
