'''
Trains a pre-training model for a univariate forecasting model.
Pre-training is done on a small portion of the training dataset.
PRE_TRAIN_RATIO is used to select the pre-training dataset
    from the training dataset.
'''
import argparse

import gc

from io import BytesIO

from keras.callbacks import Callback, ModelCheckpoint

import numpy as np

import os

from settings import TRAINING_DATA_FOLDER, PATCH_SIZE, \
    POOL_SIZE_REDUCTION, POOL_SIZE_TREND, NR_OF_BINS, \
    PRE_TRAIN_RATIO, MINI_BATCH_SIZE, \
    PROJECTION_HEAD, MASK_RATE, MSK_SCALAR, \
    NR_OF_LOOKBACK_PATCHES, NR_OF_FORECAST_PATCHES, \
    ARTIFACTS_FOLDER, NR_OF_EPOCHS, \
    NR_OF_ENCODER_BLOCKS, NR_OF_HEADS, \
    DROPOUT_RATE, ENCODER_FFN_UNITS, EMBEDDING_DIMS, \
    LEARNING_RATE, BETA_1, BETA_2, CLIP_NORM

import shutil

import sys

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from tsf_model import PreProcessor, PreTraining


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

    '''Architecture-related hyperparameters.'''
    parser.add_argument(
        '--nr_of_encoder_blocks',
        required=False,
        default=NR_OF_ENCODER_BLOCKS,
        type=int,
        help='nr_of_encoder_blocks'
    )
    parser.add_argument(
        '--nr_of_heads',
        required=False,
        default=NR_OF_HEADS,
        type=int,
        help='nr_of_heads'
    )
    parser.add_argument(
        '--dropout_rate',
        required=False,
        default=DROPOUT_RATE,
        type=float,
        help='dropout_rate'
    )
    parser.add_argument(
        '--encoder_ffn_units',
        required=False,
        default=ENCODER_FFN_UNITS,
        type=int,
        help='encoder_ffn_units'
    )
    parser.add_argument(
        '--embedding_dims',
        required=False,
        default=EMBEDDING_DIMS,
        type=int,
        help='embedding_dims'
    )
    parser.add_argument(
        '--projection_head',
        required=False,
        default=PROJECTION_HEAD,
        type=int,
        help='projection_head'
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
        '--pre_train_ratio',
        required=False,
        default=PRE_TRAIN_RATIO,
        type=float,
        help='pre_train_ratio'
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


class CustomCallback(Callback):

    def on_epoch_end(self, epoch, logs={}):
        '''
        used to stop the training when the threshold is achived.
        Also cleans the RAM.
        '''
        cos_dist = logs.get('cos_dist')
        cos_tre = logs.get('cos_tre')
        cos_sea = logs.get('cos_sea')
        if cos_dist >= 0.80 and cos_tre <= 0.80 and cos_sea <= 0.80:
            self.model.stop_training = True
            print('Stopping because threshold is achived...')

        gc.collect()


class CustomModelCheckpoint(ModelCheckpoint):
    '''Custom ModelCheckpoint by epoch frequency.'''
    def __init__(self,
                 starting_epoch_checkpoint_dir,
                 epoch_freq,
                 **kwargs):
        super().__init__(**kwargs)

        self.starting_epoch_checkpoint_dir = os.path.join(
            starting_epoch_checkpoint_dir,
            'ckpt')

        self.filepath = os.path.join(
            self.filepath,
            'ckpt')

        self.epoch_freq = epoch_freq

    def on_epoch_end(self, epoch, logs=None):
        '''Saves model and the remained epoch to resume training.'''
        if epoch % self.epoch_freq == 0:

            checkpoint_epoch = tf.train.Checkpoint(
                starting_epoch=tf.Variable(epoch, dtype=tf.int64))

            checkpoint_epoch.save(
                file_prefix=self.starting_epoch_checkpoint_dir)

            return super().on_epoch_end(epoch, logs)


def get_random_sample(lb, fc, ts, sampling_ratio):
    np.random.seed(1)
    size_of_sample = int(len(lb) * sampling_ratio)
    all_indices = np.arange(len(lb))
    np.random.shuffle(all_indices)
    rand_indices = all_indices[:size_of_sample]
    rand_indices = np.sort(rand_indices)

    return lb[rand_indices], fc[rand_indices], ts[rand_indices]


if __name__ == '__main__':
    '''
    Pre-trains a given channel.
    A training dataset should be in format of (None, timesteps).
    '''
    args = get_args()

    sChannel = input(f'Enter a channel name from {TRAINING_DATA_FOLDER}:')
    artficats_dir = os.path.join(ARTIFACTS_FOLDER, sChannel, 'pre_train')
    model_checkpoint_dir = os.path.join(artficats_dir, 'model_weights')
    starting_epoch_checkpoint_dir = os.path.join(artficats_dir,
                                                 'starting_epoch')
    tensorboard_log_dir = os.path.join(artficats_dir, 'tboard_logs')

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

    lb_train, fc_train, ts_train = get_random_sample(
        lb=lb_train,
        fc=fc_train,
        ts=ts_train,
        sampling_ratio=args.pre_train_ratio)

    oPreProcessor = PreProcessor(
        iPatchSize=PATCH_SIZE,
        iPoolSizeReduction=POOL_SIZE_REDUCTION,
        iPoolSizeTrend=POOL_SIZE_TREND,
        iNrOfBins=NR_OF_BINS
    )
    dist, tre, sea = oPreProcessor((lb_train, fc_train))
    ts_train = oPreProcessor.batch_normalizer(ts_train, training=True)

    ds_train = tf.data.Dataset.from_tensor_slices(
        (dist, tre, sea, ts_train)).batch(
            args.mini_batch_size).prefetch(tf.data.AUTOTUNE)

    model = PreTraining(
        iNrOfEncoderBlocks=args.nr_of_encoder_blocks,
        iNrOfHeads=args.nr_of_heads,
        fDropoutRate=args.dropout_rate,
        iEncoderFfnUnits=args.encoder_ffn_units,
        iEmbeddingDims=args.embedding_dims,
        iProjectionHeadUnits=args.projection_head,
        iReducedDims=tre.shape[2],
        fMskRate=MASK_RATE,
        fMskScalar=MSK_SCALAR,
        iNrOfBins=NR_OF_BINS,
        iNrOfLookbackPatches=NR_OF_LOOKBACK_PATCHES,
        iNrOfForecastPatches=NR_OF_FORECAST_PATCHES,
        tensorboard_log_dir=tensorboard_log_dir)

    mae_optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        clipnorm=args.clip_norm)

    cl_optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        clipnorm=args.clip_norm)

    model.compile(
        masked_autoencoder_optimizer=mae_optimizer,
        contrastive_optimizer=cl_optimizer)

    starting_epoch = 0

    if args.resume_training is True:
        latest_model_checkpoint = tf.train.latest_checkpoint(
            model_checkpoint_dir)
        if latest_model_checkpoint:
            model.load_weights(latest_model_checkpoint)

        latest_starting_epoch_checkpoint = tf.train.latest_checkpoint(
            starting_epoch_checkpoint_dir)

        if latest_starting_epoch_checkpoint:
            checkpoint_starting_epoch = tf.train.Checkpoint(
                starting_epoch=tf.Variable(starting_epoch, dtype=tf.int64))

            checkpoint_starting_epoch.restore(latest_starting_epoch_checkpoint)
            starting_epoch = checkpoint_starting_epoch.starting_epoch.numpy()
    else:
        shutil.rmtree(artficats_dir, ignore_errors=True)
        os.makedirs(artficats_dir)

    checkpoint_callback = CustomModelCheckpoint(
        starting_epoch_checkpoint_dir=starting_epoch_checkpoint_dir,
        filepath=model_checkpoint_dir,
        epoch_freq=3,
        save_weights_only=True,
        save_best_only=True,
        monitor='loss_mpp',
        mode='min',
        save_freq='epoch')

    custom_callback = CustomCallback()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        write_graph=True,
        write_images=False,
        histogram_freq=1)

    print(f'tensorboard --logdir=".{tensorboard_log_dir}" --bind_all')
    history = model.fit(
        ds_train,
        epochs=args.nr_of_epochs,
        verbose=2,
        initial_epoch=starting_epoch,
        shuffle=False,
        callbacks=[custom_callback, tensorboard_callback, checkpoint_callback])

    model.save(
        artficats_dir,
        overwrite=True,
        save_format='tf')

    print('Training completed.')
