import argparse

import sys


def get_args():
    '''Parses the args to build datasets.'''

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dataset_id',
        default='etth1',
        required=False,
        type=str,
        help='input_dataset_id')

    parser.add_argument(
        '--output_dataset_id',
        default='debug_purpose',
        required=False,
        type=str,
        help='output_dataset_id')

    parser.add_argument(
        '--forecast_horizon',
        default=96,
        required=False,
        type=int,
        help='forecast_horizon')

    parser.add_argument(
        '--lookback_horizon',
        default=384,
        required=False,
        type=int,
        help='lookback_horizon')

    parser.add_argument(
        '--features',
        default='S',
        required=False,
        type=str,
        help='features')

    try:
        args = parser.parse_args()
    except:  # noqa: E722
        parser.print_help()
        sys.exit(0)

    return args
