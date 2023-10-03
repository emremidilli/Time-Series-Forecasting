#!/bin/bash

channel=$1
resume_training=$2
nr_of_epochs=$3
trainable_encoder=$4
validation_rate=$5

learning_rate=0.00001
clip_norm=1.0
mini_batch_size=64

l1_ratio=0.50
alpha_regulizer=0.10

cd ../app_training/

main() {
    echo "starting to fine-tune " $channel

    docker-compose run --rm app_training \
        fine_tune.py \
        --channel=$channel \
        --resume_training=$resume_training \
        --learning_rate=$learning_rate \
        --clip_norm=$clip_norm \
        --mini_batch_size=$mini_batch_size \
        --nr_of_epochs=$nr_of_epochs \
        --validation_rate=$validation_rate \
        --alpha_regulizer=$alpha_regulizer \
        --l1_ratio=$l1_ratio \
        --trainable_encoder=$trainable_encoder

    echo "fine-tuning is successfull for " $channel
}

main

