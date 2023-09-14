import argparse

import json

import os

import sys


def get_pre_training_args():
    '''
    Parses the args for pre-training.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--channel',
        required=False,
        default='EURUSD',
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
        '--clip_norm',
        required=False,
        default=1.0,
        type=float,
        help='clip_norm'
    )
    parser.add_argument(
        '--warmup_steps',
        required=False,
        default=4000,
        type=int,
        help='warmup_steps'
    )
    parser.add_argument(
        '--scale_factor',
        required=False,
        default=1.0,
        type=float,
        help='scale_factor'
    )

    # Architecture-related hyperparameters.
    parser.add_argument(
        '--nr_of_encoder_blocks',
        required=False,
        default=1,
        type=int,
        help='nr_of_encoder_blocks'
    )
    parser.add_argument(
        '--nr_of_heads',
        required=False,
        default=4,
        type=int,
        help='nr_of_heads'
    )
    parser.add_argument(
        '--encoder_ffn_units',
        required=False,
        default=16,
        type=int,
        help='encoder_ffn_units'
    )
    parser.add_argument(
        '--embedding_dims',
        required=False,
        default=16,
        type=int,
        help='embedding_dims'
    )
    parser.add_argument(
        '--projection_head',
        required=False,
        default=8,
        type=int,
        help='projection_head'
    )
    parser.add_argument(
        '--dropout_rate',
        required=False,
        default=0.10,
        type=float,
        help='dropout_rate'
    )

    # Training-related hyperparameters
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
        default=200,
        type=int,
        help='nr_of_epochs'
    )
    parser.add_argument(
        '--pre_train_ratio',
        required=False,
        default=0.05,
        type=float,
        help='pre_train_ratio'
    )

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args


def get_fine_tuning_args():
    '''
    Parses the args.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--channel',
        required=False,
        default='EURUSD',
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


def get_data_format_config(folder_path):
    '''returns dictionary of dataformat config from datasets folder.'''
    file_path = os.path.join(folder_path, 'config.json')
    with open(file_path, 'r') as j:
        contents = json.loads(j.read())

    return contents
