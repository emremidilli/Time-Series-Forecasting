import argparse

import json

import os

import sys


def get_input_args_pre_training():
    '''Parses the args for input of pre-training task.'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--channel',
        required=False,
        default='EURUSD',
        type=str,
        help='channel')

    parser.add_argument(
        '--patch_size',
        required=False,
        default=30,
        type=int,
        help='patch_size')

    parser.add_argument(
        '--pool_size_reduction',
        required=False,
        default=5,
        type=int,
        help='pool_size_reduction')

    parser.add_argument(
        '--pool_size_trend',
        required=False,
        default=2,
        type=int,
        help='pool_size_trend')

    parser.add_argument(
        '--nr_of_bins',
        required=False,
        default=8,
        type=int,
        help='nr_of_bins')

    parser.add_argument(
        '--pre_train_ratio',
        required=False,
        default=0.05,
        type=float,
        help='pre_train_ratio')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args


def get_input_args_fine_tuning():
    '''Parses the args for input of fine_tuning task.'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--channel',
        required=False,
        default='EURUSD',
        type=str,
        help='channel')

    parser.add_argument(
        '--patch_size',
        required=False,
        default=30,
        type=int,
        help='patch_size')

    parser.add_argument(
        '--pool_size_reduction',
        required=False,
        default=5,
        type=int,
        help='pool_size_reduction')

    parser.add_argument(
        '--pool_size_trend',
        required=False,
        default=2,
        type=int,
        help='pool_size_trend')

    parser.add_argument(
        '--nr_of_bins',
        required=False,
        default=8,
        type=int,
        help='nr_of_bins')

    parser.add_argument(
        '--quantiles',
        required=False,
        default="[0.10, 0.50, 0.90]",
        type=eval,
        help='quantiles')

    parser.add_argument(
        '--mask_scalar',
        required=False,
        default=0.53,
        type=float,
        help='mask_scalar')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args


def get_input_args_inference():
    '''Parses the args for input of inference task.'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--lb_dir',
        required=False,
        type=str,
        help='lb_dir')

    parser.add_argument(
        '--ts_dir',
        required=False,
        type=str,
        help='ts_dir')

    parser.add_argument(
        '--pre_processor_dir',
        required=False,
        type=str,
        help='pre_processor_dir')

    parser.add_argument(
        '--save_dir',
        required=False,
        type=str,
        help='save_dir')

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
