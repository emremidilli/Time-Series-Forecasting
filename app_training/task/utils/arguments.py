import argparse

import sys


def get_pre_training_args():
    '''Parses the args for pre-training.'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_id',
        required=False,
        default='model_05',
        type=str,
        help='model_id')

    parser.add_argument(
        '--dataset_id',
        required=False,
        default='dataset_03',
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
        default=32,
        type=int,
        help='encoder_ffn_units')
    parser.add_argument(
        '--embedding_dims',
        required=False,
        default=32,
        type=int,
        help='embedding_dims')
    parser.add_argument(
        '--projection_head',
        required=False,
        default=32,
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
        default=128,
        type=int,
        help='mini_batch_size')

    parser.add_argument(
        '--nr_of_epochs',
        required=False,
        default=20,
        type=int,
        help='nr_of_epochs')
    parser.add_argument(
        '--mask_scalar',
        required=False,
        default=0.001,
        type=float,
        help='mask_scalar')
    parser.add_argument(
        '--mask_rate',
        required=False,
        default=0.70,
        type=float,
        help='mask_rate')

    parser.add_argument(
        '--validation_rate',
        required=False,
        default=0.15,
        type=float,
        help='validation_rate')

    parser.add_argument(
        '--mae_threshold_comp',
        required=False,
        default=0.1,
        type=float,
        help='mae_threshold_comp')

    parser.add_argument(
        '--mae_threshold_tre',
        required=False,
        default=0.1,
        type=float,
        help='mae_threshold_tre')

    parser.add_argument(
        '--mae_threshold_sea',
        required=False,
        default=0.1,
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
        default=24,
        type=int,
        help='patch_size')

    parser.add_argument(
        '--lookback_coefficient',
        required=False,
        default=2,
        type=int,
        help='lookback_coefficient')

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
        default='model_06',
        type=str,
        help='model_id')

    parser.add_argument(
        '--pre_trained_model_id',
        required=False,
        default='model_05',
        type=str,
        help='pre_trained_model_id')

    parser.add_argument(
        '--dataset_id',
        required=False,
        default='dataset_04',
        type=str,
        help='dataset_id')

    parser.add_argument(
        '--resume_training',
        required=False,
        default='N',
        choices=['Y', 'N'],
        type=str,
        help='resume_training')

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

    parser.add_argument(
        '--alpha_regulizer',
        required=False,
        default=0.10,
        type=float,
        help='alpha_regulizer')

    parser.add_argument(
        '--l1_ratio',
        required=False,
        default=0.10,
        type=float,
        help='l1_ratio')

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

    parser.add_argument(
        '--validation_rate',
        required=False,
        default=0.15,
        type=float,
        help='validation_rate')

    parser.add_argument(
        '--nr_of_layers',
        required=False,
        default=1,
        type=int,
        help='nr_of_layers')

    parser.add_argument(
        '--hidden_dims',
        required=False,
        default=128,
        type=int,
        help='hidden_dims')

    parser.add_argument(
        '--nr_of_heads',
        required=False,
        default=6,
        type=int,
        help='nr_of_heads')

    parser.add_argument(
        '--dropout_rate',
        required=False,
        default=0.10,
        type=float,
        help='dropout_rate')

    parser.add_argument(
        '--pre_trained_lookback_coefficient',
        required=False,
        default=2,
        type=int,
        help='pre_trained_lookback_coefficient')

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
        '--input_dataset_dir',
        required=False,
        default="./tsf-bin/05_inference/ETTh1/input/",
        type=str,
        help='input_dataset_dir')

    parser.add_argument(
        '--model_dir',
        required=False,
        default="./tsf-bin/04_artifacts/ETTh1/fine_tune/saved_model/",
        type=str,
        help='model_dir')

    parser.add_argument(
        '--output_save_dir',
        required=False,
        default="./tsf-bin/05_inference/ETTh1/output/",
        type=str,
        help='output_save_dir')

    parser.add_argument(
        '--nr_of_forecasting_steps',
        required=False,
        default=168,
        type=int,
        help='nr_of_forecasting_steps')

    parser.add_argument(
        '--begin_scalar',
        required=False,
        default=0.50,
        type=float,
        help='begin_scalar')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args
