import argparse

import sys


def get_args_to_build_datasets():
    '''Parses the args to build datasets.'''

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--channel',
        required=True,
        type=str,
        help='channel')

    parser.add_argument(
        '--target_group',
        required=True,
        type=str,
        help='target_group')

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
        type=int,
        help='test_size')

    try:
        args = parser.parse_args()
    except:  # noqa: E722
        parser.print_help()
        sys.exit(0)

    return args


def get_args_to_build_date_features():
    '''Parses the args to build date features.'''

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--channel',
        required=True,
        type=str,
        help='channel')

    parser.add_argument(
        '--datetime_features',
        required=True,
        default="['month', 'day', 'dayofweek', 'hour', 'minute']",
        type=eval,
        help='datetime_features')

    try:
        args = parser.parse_args()
    except:  # noqa: E722
        parser.print_help()
        sys.exit(0)

    return args
