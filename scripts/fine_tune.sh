#!/bin/bash

model_id="model_ft_ETTm1_720_384_20240310"
pre_trained_model_id="model_pt_ETTm1_20240310"
dataset_id="ds_universal_ETTm1_720_384_M_ft"
resume_training="N"
concat_train_val="N"
patience=5
mini_batch_size=128
learning_rate=0.0001
clip_norm=1.0
nr_of_epochs=1000

cd ../app_training/

main() {
    echo "starting to fine-tune " $model_id

    docker-compose run --rm app_training \
        fine_tune.py \
        --model_id=$model_id \
        --pre_trained_model_id=$pre_trained_model_id \
        --dataset_id=$dataset_id \
        --resume_training=$resume_training \
        --concat_train_val=$concat_train_val \
        --patience=$patience \
        --mini_batch_size=$mini_batch_size \
        --learning_rate=$learning_rate \
        --clip_norm=$clip_norm \
        --nr_of_epochs=$nr_of_epochs

    echo "fine-tuning is successfull for " $model_id
}

main
