#!/bin/bash

channel=$1
resume_training=$2
nr_of_epochs=$3

clip_norm=1.0
warmup_steps=1000
scale_factor=0.10
nr_of_encoder_blocks=4
nr_of_heads=4
encoder_ffn_units=128
embedding_dims=128
projection_head=32
dropout_rate=0.10
mini_batch_size=64
pre_train_ratio=0.25

cd ../app_training/

echo "starting to pre-train" $channel

docker-compose run --rm app_training \
    fine_tune.py \
    --channel=$channel \
    --resume_training=$resume_training \
    --nr_of_epochs=$nr_of_epochs \
    --clip_norm=$clip_norm \
    --warmup_steps=$warmup_steps \
    --scale_factor=$scale_factor \
    --nr_of_encoder_blocks=$nr_of_encoder_blocks \
    --nr_of_heads=$nr_of_heads \
    --encoder_ffn_units=$encoder_ffn_units \
    --embedding_dims=$embedding_dims \
    --projection_head=$projection_head \
    --dropout_rate=$dropout_rate \
    --mini_batch_size=$mini_batch_size \
    --pre_train_ratio=$pre_train_ratio

echo "pre-train is successfull for " $channel
