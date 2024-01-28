#!/bin/bash

model_id="model_03"
dataset_id="dataset_03"
resume_training="Y"
nr_of_epochs=35000
mask_rate=0.40
mask_scalar=0.001
validation_rate=0.15
mini_batch_size=128
mae_threshold_tre=0.10
mae_threshold_sea=0.001
mae_threshold_comp=0.50
cl_threshold=0.25
save_model="Y"
patch_size=24
cl_margin=0.25
lookback_coefficient=2

cd ../app_training/

main() {
    clip_norm=$1 # 1.0
    warmup_steps=$2 # 1000
    scale_factor=$3 # 0.10
    nr_of_encoder_blocks=$4 # 1
    nr_of_heads=$5 # 1
    encoder_ffn_units=$6 # 256
    embedding_dims=$7 # 256
    projection_head=$8 # 16
    dropout_rate=$9 # 0.10

    echo "starting to pre-train"

    docker-compose run --rm app_training \
        pre_train.py \
        --model_id=$model_id \
        --dataset_id=$dataset_id \
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
        --validation_rate=$validation_rate \
        --mae_threshold_tre=$mae_threshold_tre \
        --mae_threshold_sea=$mae_threshold_sea \
        --mae_threshold_comp=$mae_threshold_comp \
        --cl_threshold=$cl_threshold \
        --cl_margin=$cl_margin \
        --save_model=$save_model \
        --patch_size=$patch_size \
        --lookback_coefficient=$lookback_coefficient

    echo "pre-training is completed"

}

main 1.0 4000 1.0 6 6 128 128 32 0.10