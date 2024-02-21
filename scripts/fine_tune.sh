#!/bin/bash

model_id="model_ft_20240218_01"
pre_trained_model_id="model_pt_20240218_01"
dataset_id="ds_universal_ETTh1_96_4_M_ft"
resume_training="N"
validation_rate=0.00
mini_batch_size=128
learning_rate=0.0001
clip_norm=1.0
nr_of_epochs=200
fine_tune_backbone="Y"

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
