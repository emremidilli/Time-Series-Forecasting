import argparse

import json

import os

import sys


def get_args_to_build_datasets():
    '''Parses the args to build datasets.'''

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_id',
        required=True,
        type=str,
        help='model_id')

    parser.add_argument(
        '--dataset_id',
        required=True,
        type=str,
        help='dataset_id')

    parser.add_argument(
        '--list_of_covariates',
        required=True,
        type=eval,
        help='list_of_covariates')

    parser.add_argument(
        '--forecast_horizon',
        required=True,
        type=int,
        help='forecast_horizon')

    parser.add_argument(
        '--lookback_coefficient',
        required=True,
        type=int,
        help='lookback_coefficient')

    parser.add_argument(
        '--step_size',
        required=True,
        type=int,
        help='step_size')

    parser.add_argument(
        '--test_size',
        required=True,
        type=float,
        help='test_size')

    parser.add_argument(
        '--raw_frequency',
        required=True,
        type=str,
        help='raw_frequency')

    parser.add_argument(
        '--datetime_features',
        required=True,
        type=eval,
        help='datetime_features')

    try:
        args = parser.parse_args()
    except:  # noqa: E722
        parser.print_help()
        sys.exit(0)

    return args


def save_config_file(folder_dir, args):
    '''
    checks if there is already a config file \
    if so, it saves the received arguments as a json file.
    if not, it generates a new config file.
    '''

    dic_args = vars(args)
    file_path = os.path.join(folder_dir, 'config.json')

    if os.path.exists(file_path) is True:
        with open(file_path, 'r') as j:
            contents = json.loads(j.read())
            dic_args.update(contents)

    json_object = json.dumps(dic_args, indent=4)

    with open(file=file_path, mode='w') as output_file:
        output_file.write(json_object)
