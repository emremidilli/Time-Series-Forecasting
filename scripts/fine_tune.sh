#!/bin/bash

channel=$1
resume_training=$2
nr_of_epochs=$3

learning_rate=0.0001
clip_norm=1.0
mini_batch_size=64
validation_rate=0.15

cd ../app_training/

echo "starting to fine-tune " $channel

docker-compose run --rm app_training \
    fine_tune.py \
    --channel=$channel \
    --resume_training=$resume_training \
    --learning_rate=$learning_rate \
    --clip_norm=$clip_norm \
    --mini_batch_size=$mini_batch_size \
    --nr_of_epochs=$nr_of_epochs \
    --validation_rate=$validation_rate

echo "fine-tuning is successfull for " $channel