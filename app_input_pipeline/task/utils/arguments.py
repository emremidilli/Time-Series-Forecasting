import argparse

import json

import os

import sys


def get_input_args_pre_training():
    '''Parses the args for input of pre-training task.'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_id',
        required=False,
        default='dataset_03',
        type=str,
        help='dataset_id')

    parser.add_argument(
        '--pool_size_trend',
        required=False,
        default=24,
        type=int,
        help='pool_size_trend')

    parser.add_argument(
        '--sigma',
        required=False,
        default=3,
        type=float,
        help='sigma')

    parser.add_argument(
        '--scale_data',
        required=False,
        choices=['Y', 'N'],
        default='Y',
        type=str,
        help='scale_data')

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
        '--dataset_id',
        required=False,
        default='dataset_03',
        type=str,
        help='dataset_id')

    parser.add_argument(
        '--pool_size_trend',
        required=False,
        default=24,
        type=int,
        help='pool_size_trend')

    parser.add_argument(
        '--sigma',
        required=False,
        default=3,
        type=float,
        help='sigma')

    parser.add_argument(
        '--scale_data',
        required=False,
        choices=['Y', 'N'],
        default='Y',
        type=str,
        help='scale_data')

    parser.add_argument(
        '--begin_scalar',
        required=False,
        default=0.50,
        type=float,
        help='begin_scalar')

    parser.add_argument(
        '--end_scalar',
        required=False,
        default=0.50,
        type=float,
        help='end_scalar')

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
        default=os.path.join(
            './tsf-bin',
            os.environ['FORMATTED_NAME'],
            'ETTh1',
            'lb_test.npy'),
        type=str,
        help='lb_dir')

    parser.add_argument(
        '--ts_dir',
        required=False,
        default=os.path.join(
            './tsf-bin',
            os.environ['FORMATTED_NAME'],
            'ETTh1',
            'ts_test.npy'),
        type=str,
        help='ts_dir')

    parser.add_argument(
        '--pre_processor_dir',
        required=False,
        default=os.path.join(
            './tsf-bin',
            os.environ['PREPROCESSED_NAME'],
            'ETTh1',
            'fine_tune',
            'input_preprocessor'),
        type=str,
        help='pre_processor_dir')

    parser.add_argument(
        '--save_dir',
        required=False,
        default=os.path.join(
            './tsf-bin',
            os.environ['INFERENCE_NAME'],
            'ETTh1',
            'input'),
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
