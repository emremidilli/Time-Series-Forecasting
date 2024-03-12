#!/bin/bash

model_id="model_pt_ETTm2_20240310"
dataset_id="ds_universal_ETTm2_96_4_S_pt"
resume_training="N"
concat_train_val="N"
patience=5
nr_of_epochs=1500
mask_rate=0.40
mask_scalar=0.00
mini_batch_size=128
mae_threshold_comp=0.20
mae_threshold_tre=0.20
mae_threshold_sea=0.005
save_model="Y"
patch_size=16
cl_margin=0.25
lookback_coefficient=4
prompt_pool_size=30
nr_of_most_similar_prompts=3

nr_of_encoder_blocks=3
nr_of_heads=4
clip_norm=1.0
warmup_steps=4000
scale_factor=1.0
encoder_ffn_units=128
embedding_dims=16
projection_head=16
dropout_rate=0.30

cd ../app_training/

main() {

    echo "starting to pre-train"

    docker-compose run --rm app_training \
        pre_train.py \
        --model_id=$model_id \
        --dataset_id=$dataset_id \
        --resume_training=$resume_training \
        --concat_train_val=$concat_train_val \
        --patience=$patience \
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
        --mae_threshold_tre=$mae_threshold_tre \
        --mae_threshold_sea=$mae_threshold_sea \
        --mae_threshold_comp=$mae_threshold_comp \
        --cl_margin=$cl_margin \
        --save_model=$save_model \
        --patch_size=$patch_size \
        --lookback_coefficient=$lookback_coefficient \
        --prompt_pool_size=$prompt_pool_size \
        --nr_of_most_similar_prompts=$nr_of_most_similar_prompts

    echo "pre-training is completed"

}

main