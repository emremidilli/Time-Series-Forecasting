#!/bin/bash

model_id=$1
resume_training=$2
nr_of_epochs=$3

clip_norm=1.0
warmup_steps=1000
scale_factor=0.10
nr_of_encoder_blocks=1
nr_of_heads=1
encoder_ffn_units=256
embedding_dims=256
projection_head=16
dropout_rate=0.10
mini_batch_size=128
mask_rate=0.70
mask_scalar=0.001
validation_rate=0.15

cd ../app_training/

main() {

    echo "starting to pre-train" $model_id

    docker-compose run --rm app_training \
        pre_train.py \
        --model_id=$model_id \
        --resume_training=$resume_training \
        --nr_of_epochs=$nr_of_epochs \
        --mask_rate=$mask_rate \
        --mask_scalar=$mask_scalar \
        --mini_batch_size=$mini_batch_size \
        --clip_norm=$clip_norm \
        --nr_of_encoder_blocks=$nr_of_encoder_blocks \
        --nr_of_heads=$nr_of_heads \
        --dropout_rate=$dropout_rate \
        --encoder_ffn_units=$encoder_ffn_units \
        --embedding_dims=$embedding_dims \
        --projection_head=$projection_head \
        --warmup_steps=$warmup_steps \
        --scale_factor=$scale_factor \
        --validation_rate=$validation_rate

    echo "pre-training is successfull for " $model_id

}

main
