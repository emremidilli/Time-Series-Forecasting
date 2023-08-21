import argparse

from io import BytesIO

import numpy as np

import os

from settings import TRAINING_DATA_FOLDER, PATCH_SIZE, \
    POOL_SIZE_REDUCTION, POOL_SIZE_TREND, NR_OF_BINS, \
    ARTIFACTS_FOLDER, QUANTILES, LEARNING_RATE, BETA_1, \
    BETA_2, CLIP_NORM, MINI_BATCH_SIZE, NR_OF_EPOCHS

import sys

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from tsf_model import InputPreProcessor, TargetPreProcessor  # , FineTuning


def get_args():
    '''
    Parses the args.
    '''
    parser = argparse.ArgumentParser()

    '''Optimizer-related hyperparameters.'''
    parser.add_argument(
        '--learning_rate',
        required=False,
        default=LEARNING_RATE,
        type=float,
        help='learning_rate'
    )
    parser.add_argument(
        '--beta_1',
        required=False,
        default=BETA_1,
        type=float,
        help='beta_1'
    )
    parser.add_argument(
        '--beta_2',
        required=False,
        default=BETA_2,
        type=float,
        help='beta_2'
    )
    parser.add_argument(
        '--clip_norm',
        required=False,
        default=CLIP_NORM,
        type=float,
        help='clip_norm'
    )

    '''Training-related hyperparameters'''
    parser.add_argument(
        '--mini_batch_size',
        required=False,
        default=MINI_BATCH_SIZE,
        type=int,
        help='mini_batch_size'
    )

    parser.add_argument(
        '--nr_of_epochs',
        required=False,
        default=NR_OF_EPOCHS,
        type=int,
        help='nr_of_epochs'
    )

    parser.add_argument(
        '--resume_training',
        required=False,
        default='False',
        choices=[True, False],
        type=eval,
        help='resume_training'
    )

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    print(args)

    return args


if __name__ == '__main__':
    '''
    Fine tunes a given channel.
    A training dataset should be in format of (None, timesteps).
    '''
    args = get_args()

    sChannel = input(f'Enter a channel name from {TRAINING_DATA_FOLDER}:')
    artifacts_dir = os.path.join(ARTIFACTS_FOLDER, sChannel, 'fine_tune')
    pre_trained_saved_model_dir = os.path.join(ARTIFACTS_FOLDER,
                                               sChannel,
                                               'pre_train',
                                               'saved_model')

    model_checkpoint_dir = os.path.join(artifacts_dir, 'model_weights')
    starting_epoch_checkpoint_dir = os.path.join(artifacts_dir,
                                                 'starting_epoch')
    tensorboard_log_dir = os.path.join(artifacts_dir, 'tboard_logs')

    lb_train = np.load(
        BytesIO(
            file_io.read_file_to_string(
                f'{TRAINING_DATA_FOLDER}/{sChannel}/lb_train.npy',
                binary_mode=True)))

    fc_train = np.load(
        BytesIO(
            file_io.read_file_to_string(
                f'{TRAINING_DATA_FOLDER}/{sChannel}/fc_train.npy',
                binary_mode=True)))

    ts_train = np.load(
        BytesIO(
            file_io.read_file_to_string(
                f'{TRAINING_DATA_FOLDER}/{sChannel}/ts_train.npy',
                binary_mode=True)))

    input_pre_processor = InputPreProcessor(
        iPatchSize=PATCH_SIZE,
        iPoolSizeReduction=POOL_SIZE_REDUCTION,
        iPoolSizeTrend=POOL_SIZE_TREND,
        iNrOfBins=NR_OF_BINS
    )
    target_pre_processor = TargetPreProcessor(iPatchSize=PATCH_SIZE,
                                              quantiles=QUANTILES)

    dist, tre, sea = input_pre_processor((lb_train, fc_train))
    qntl = target_pre_processor((lb_train, fc_train))
    ts_train = input_pre_processor.batch_normalizer(ts_train, training=True)

    ds_train = tf.data.Dataset.from_tensor_slices(
        (dist, tre, sea, qntl, ts_train)).batch(
            args.mini_batch_size).prefetch(tf.data.AUTOTUNE)

    '''
    load the previously saved pre-trainined model.
    '''

    # model = FineTuning(
    #     ...,
    #     nr_of_quantiles=len(QUANTILES))
