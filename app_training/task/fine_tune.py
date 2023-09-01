import argparse

from io import BytesIO

import numpy as np

import os

from settings import TRAINING_DATA_FOLDER, PATCH_SIZE, \
    POOL_SIZE_REDUCTION, POOL_SIZE_TREND, NR_OF_BINS, \
    ARTIFACTS_FOLDER, QUANTILES, NR_OF_FORECAST_PATCHES, MSK_SCALAR

import shutil

import sys

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from tsf_model import InputPreProcessor, TargetPreProcessor, FineTuning

from utils import CustomModelCheckpoint


def get_args():
    '''
    Parses the args.
    '''
    parser = argparse.ArgumentParser()

    '''Optimizer-related hyperparameters.'''
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

    '''Training-related hyperparameters'''
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
        '--resume_training',
        required=False,
        default='False',
        choices=[True, False],
        type=eval,
        help='resume_training'
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

    print(args)

    return args


def get_pre_trained_representation(channel):
    pre_trained_model_dir = os.path.join(ARTIFACTS_FOLDER,
                                         channel,
                                         'pre_train',
                                         'saved_model')

    pre_trained_model = tf.keras.models.load_model(pre_trained_model_dir)

    con_temp_pret = pre_trained_model.encoder_representation

    return con_temp_pret


def train_test_split(ds, test_rate=0.15):
    '''Splits tf.data.Dataset to train and test datasets.'''
    nr_of_samples = ds.cardinality().numpy()

    train_size = int(nr_of_samples * (1 - test_rate))

    ds_train = ds.take(train_size)
    ds_test = ds.skip(train_size)

    return ds_train, ds_test


if __name__ == '__main__':
    '''
    Fine tunes a given channel.
    A training dataset should be in format of (None, timesteps).
    '''
    args = get_args()

    channel = input(f'Enter a channel name from {TRAINING_DATA_FOLDER}:')

    artifacts_dir = os.path.join(ARTIFACTS_FOLDER, channel, 'fine_tune')
    model_checkpoint_dir = os.path.join(artifacts_dir, 'model_weights')
    saved_model_dir = os.path.join(artifacts_dir, 'saved_model')
    starting_epoch_checkpoint_dir = os.path.join(artifacts_dir,
                                                 'starting_epoch')
    tensorboard_log_dir = os.path.join(artifacts_dir, 'tboard_logs')

    lb_train = np.load(
        BytesIO(
            file_io.read_file_to_string(
                f'{TRAINING_DATA_FOLDER}/{channel}/lb_train.npy',
                binary_mode=True)))

    fc_train = np.load(
        BytesIO(
            file_io.read_file_to_string(
                f'{TRAINING_DATA_FOLDER}/{channel}/fc_train.npy',
                binary_mode=True)))

    ts_train = np.load(
        BytesIO(
            file_io.read_file_to_string(
                f'{TRAINING_DATA_FOLDER}/{channel}/ts_train.npy',
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
    dist, tre, sea = input_pre_processor.mask_forecast_patches(
        inputs=(dist, tre, sea),
        nr_of_patches=NR_OF_FORECAST_PATCHES,
        msk_scalar=MSK_SCALAR
    )
    qntl = target_pre_processor((lb_train, fc_train))
    ts = input_pre_processor.batch_normalizer(ts_train, training=True)

    ds = tf.data.Dataset.from_tensor_slices(((dist, tre, sea, ts), qntl))
    ds_train = ds
    ds_val = None
    if args.validation_rate > 0:
        ds_train, ds_val = train_test_split(ds, test_rate=args.validation_rate)
        ds_val = ds_val.batch(args.mini_batch_size).prefetch(tf.data.AUTOTUNE)

    ds_train = ds_train.batch(args.mini_batch_size).prefetch(tf.data.AUTOTUNE)

    con_temp_pret = get_pre_trained_representation(channel=channel)

    model = FineTuning(
        con_temp_pret=con_temp_pret,
        nr_of_time_steps=NR_OF_FORECAST_PATCHES,
        nr_of_quantiles=len(QUANTILES))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        clipnorm=args.clip_norm)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(name='mse'),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.CosineSimilarity(name='cos')
        ]
    )

    checkpoint_callback = CustomModelCheckpoint(
        starting_epoch_checkpoint_dir=starting_epoch_checkpoint_dir,
        filepath=model_checkpoint_dir,
        epoch_freq=3,
        save_weights_only=True,
        save_best_only=True,
        monitor='loss',
        mode='min',
        save_freq='epoch')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        write_graph=True,
        write_images=False,
        histogram_freq=1)

    terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()

    starting_epoch = 0
    if args.resume_training is True:
        starting_epoch = checkpoint_callback.\
            get_most_recent_weight_and_epoch_nr(model=model)
    else:
        shutil.rmtree(artifacts_dir, ignore_errors=True)
        os.makedirs(artifacts_dir)

    print(f'tensorboard --logdir=".{tensorboard_log_dir}" --bind_all')
    model.fit(
        ds_train,
        epochs=args.nr_of_epochs,
        verbose=2,
        validation_data=ds_val,
        initial_epoch=starting_epoch,
        shuffle=False,
        callbacks=[
            terminate_on_nan_callback,
            tensorboard_callback,
            checkpoint_callback])

    model.save(
        saved_model_dir,
        overwrite=True,
        save_format='tf')

    print('Training completed.')
