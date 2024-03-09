import argparse

import sys


def get_pre_training_args():
    '''Parses the args for pre-training.'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_id',
        required=False,
        default='test_pt',
        type=str,
        help='model_id')

    parser.add_argument(
        '--dataset_id',
        required=False,
        default='test_pt',
        type=str,
        help='dataset_id')

    parser.add_argument(
        '--resume_training',
        required=False,
        default='N',
        choices=['Y', 'N'],
        type=str,
        help='resume_training')

    parser.add_argument(
        '--concat_train_val',
        required=False,
        default='N',
        choices=['Y', 'N'],
        type=str,
        help='concat_train_val')

    parser.add_argument(
        '--save_model',
        required=False,
        default='Y',
        choices=['Y', 'N'],
        type=str,
        help='save_model')

    # Optimizer-related hyperparameters.
    parser.add_argument(
        '--clip_norm',
        required=False,
        default=1.0,
        type=float,
        help='clip_norm')
    parser.add_argument(
        '--warmup_steps',
        required=False,
        default=4000,
        type=int,
        help='warmup_steps')
    parser.add_argument(
        '--scale_factor',
        required=False,
        default=1.0,
        type=float,
        help='scale_factor')

    # Architecture-related hyperparameters.
    parser.add_argument(
        '--nr_of_encoder_blocks',
        required=False,
        default=1,
        type=int,
        help='nr_of_encoder_blocks')
    parser.add_argument(
        '--nr_of_heads',
        required=False,
        default=1,
        type=int,
        help='nr_of_heads')
    parser.add_argument(
        '--encoder_ffn_units',
        required=False,
        default=8,
        type=int,
        help='encoder_ffn_units')
    parser.add_argument(
        '--embedding_dims',
        required=False,
        default=8,
        type=int,
        help='embedding_dims')
    parser.add_argument(
        '--projection_head',
        required=False,
        default=8,
        type=int,
        help='projection_head')
    parser.add_argument(
        '--dropout_rate',
        required=False,
        default=0.10,
        type=float,
        help='dropout_rate')

    # Training-related hyperparameters
    parser.add_argument(
        '--mini_batch_size',
        required=False,
        default=8,
        type=int,
        help='mini_batch_size')

    parser.add_argument(
        '--nr_of_epochs',
        required=False,
        default=5,
        type=int,
        help='nr_of_epochs')
    parser.add_argument(
        '--mask_scalar',
        required=False,
        default=0.00,
        type=float,
        help='mask_scalar')
    parser.add_argument(
        '--mask_rate',
        required=False,
        default=0.40,
        type=float,
        help='mask_rate')

    parser.add_argument(
        '--mae_threshold_comp',
        required=False,
        default=0.50,
        type=float,
        help='mae_threshold_comp')

    parser.add_argument(
        '--mae_threshold_tre',
        required=False,
        default=0.50,
        type=float,
        help='mae_threshold_tre')

    parser.add_argument(
        '--mae_threshold_sea',
        required=False,
        default=0.05,
        type=float,
        help='mae_threshold_sea')

    parser.add_argument(
        '--cl_threshold',
        required=False,
        default=0.25,
        type=float,
        help='cl_threshold')

    parser.add_argument(
        '--cl_margin',
        required=False,
        default=0.25,
        type=float,
        help='cl_margin')

    parser.add_argument(
        '--patch_size',
        required=False,
        default=48,
        type=int,
        help='patch_size')

    parser.add_argument(
        '--lookback_coefficient',
        required=False,
        default=1,
        type=int,
        help='lookback_coefficient')

    parser.add_argument(
        '--prompt_pool_size',
        required=False,
        default=2,
        type=int,
        help='prompt_pool_size')

    parser.add_argument(
        '--nr_of_most_similar_prompts',
        required=False,
        default=1,
        type=int,
        help='nr_of_most_similar_prompts')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args


def get_fine_tuning_args():
    '''Parses the args for fine-tuning.'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_id',
        required=False,
        default='test_ft',
        type=str,
        help='model_id')

    parser.add_argument(
        '--pre_trained_model_id',
        required=False,
        default='test_pt',
        type=str,
        help='pre_trained_model_id')

    parser.add_argument(
        '--dataset_id',
        required=False,
        default='ds_universal_ETTh2_96_4_M_ft',
        type=str,
        help='dataset_id')

    parser.add_argument(
        '--resume_training',
        required=False,
        default='N',
        choices=['Y', 'N'],
        type=str,
        help='resume_training')

    parser.add_argument(
        '--concat_train_val',
        required=False,
        default='N',
        choices=['Y', 'N'],
        type=str,
        help='concat_train_val')

    # Optimizer-related hyperparameters.
    parser.add_argument(
        '--learning_rate',
        required=False,
        default=1e-5,
        type=float,
        help='learning_rate')

    parser.add_argument(
        '--clip_norm',
        required=False,
        default=1.0,
        type=float,
        help='clip_norm')

    # Training-related hyperparameters.
    parser.add_argument(
        '--mini_batch_size',
        required=False,
        default=32,
        type=int,
        help='mini_batch_size')

    parser.add_argument(
        '--nr_of_epochs',
        required=False,
        default=900,
        type=int,
        help='nr_of_epochs')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args


def get_inference_args():
    '''Parses the args for inference.'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_id',
        required=False,
        default='model_20240203_06_ft',
        type=str,
        help='model_id')

    parser.add_argument(
        '--input_dir',
        required=False,
        default='./tsf-bin/05_inference/ds_universal_ETTh1_96_4_S/input/',
        type=str,
        help='input_dir')

    parser.add_argument(
        '--output_dir',
        required=False,
        default='./tsf-bin/05_inference/ds_universal_ETTh1_96_4_S/output/',
        type=str,
        help='output_dir')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args
