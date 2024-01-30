#!/bin/bash

model_id="model_04"
pre_trained_model_id="model_03"
dataset_id="dataset_04"
resume_training="N"
validation_rate=0.15
mini_batch_size=32
learning_rate=0.001
clip_norm=1.0
nr_of_epochs=1000
alpha_regulizer=0.1
l1_ratio=0.1
nr_of_layers=1
hidden_dims=128
nr_of_heads=6
dropout_rate=0.10


cd ../app_training/

main() {
    echo "starting to fine-tune " $model_id

    docker-compose run --rm app_training \
        fine_tune.py \
        --pre_trained_model_id=$pre_trained_model_id \
        --dataset_id=$dataset_id \
        --resume_training=$resume_training \
        --validation_rate=$validation_rate \
        --mini_batch_size=$mini_batch_size \
        --learning_rate=$learning_rate \
        --clip_norm=$clip_norm \
        --nr_of_epochs=$nr_of_epochs \
        --alpha_regulizer=$alpha_regulizer \
        --l1_ratio=$l1_ratio \
        --nr_of_layers=$nr_of_layers \
        --hidden_dims=$hidden_dims \
        --nr_of_heads=$hidden_dims \
        --dropout_rate=$dropout_rate

    echo "fine-tuning is successfull for " $model_id
}

main

