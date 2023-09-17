import argparse

import json

import os

import sys


def get_pre_training_args():
    '''Parses the args for pre-training.'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--channel',
        required=False,
        default='USDCAD',
        type=str,
        help='channel')
    parser.add_argument(
        '--resume_training',
        required=False,
        default='N',
        choices=['Y', 'N'],
        type=str,
        help='resume_training')

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
        default=4,
        type=int,
        help='nr_of_heads')
    parser.add_argument(
        '--encoder_ffn_units',
        required=False,
        default=16,
        type=int,
        help='encoder_ffn_units')
    parser.add_argument(
        '--embedding_dims',
        required=False,
        default=16,
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
        default=64,
        type=int,
        help='mini_batch_size')

    parser.add_argument(
        '--nr_of_epochs',
        required=False,
        default=200,
        type=int,
        help='nr_of_epochs')
    parser.add_argument(
        '--mask_scalar',
        required=False,
        default=0.53,
        type=float,
        help='mask_scalar')
    parser.add_argument(
        '--mask_rate',
        required=False,
        default=0.70,
        type=float,
        help='mask_rate')

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
        '--channel',
        required=False,
        default='USDCAD',
        type=str,
        help='channel'
    )

    parser.add_argument(
        '--resume_training',
        required=False,
        default='N',
        choices=['Y', 'N'],
        type=str,
        help='resume_training'
    )

    # Optimizer-related hyperparameters.
    parser.add_argument(
        '--learning_rate',
        required=False,
        default=1e-5,
        type=float,
        help='learning_rate'
    )
    parser.add_argument(
        '--clip_norm',
        required=False,
        default=1.0,
        type=float,
        help='clip_norm'
    )

    # Training-related hyperparameters.
    parser.add_argument(
        '--mini_batch_size',
        required=False,
        default=64,
        type=int,
        help='mini_batch_size'
    )
    parser.add_argument(
        '--nr_of_epochs',
        required=False,
        default=100,
        type=int,
        help='nr_of_epochs'
    )
    parser.add_argument(
        '--validation_rate',
        required=False,
        default=0.15,
        type=float,
        help='validation_rate'
    )

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
        default="./tsf-bin/05_inference/EURUSD/input/",
        type=str,
        help='input_dataset_dir'
    )

    parser.add_argument(
        '--model_dir',
        required=False,
        default="./tsf-bin/04_artifacts/EURUSD/fine_tune/saved_model/",
        type=str,
        help='model_dir'
    )

    parser.add_argument(
        '--output_save_dir',
        required=False,
        default="./tsf-bin/05_inference/EURUSD/output/",
        type=str,
        help='output_save_dir'
    )

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args


def get_data_format_config(folder_path):
    '''returns dictionary of dataformat config from datasets folder.'''
    file_path = os.path.join(folder_path, 'config.json')
    with open(file_path, 'r') as j:
        contents = json.loads(j.read())

    return contents
