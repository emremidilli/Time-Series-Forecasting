'''
    Trains a pre-training model for a univariate forecasting model.
    Pre-training is done on training dataset.
    But, it is performed on a small portion of the training dataset.
    PRE_TRAIN_RATIO is used to select the pre-training dataset out of training
        dataset randomly.

    inputs:
        lb_train: (None, timesteps)
        fc_train: (None, timesteps)
'''
import argparse

import numpy as np

import os

import shutil

from settings import TRAINING_DATA_FOLDER, PATCH_SIZE, \
    PATCH_SAMPLE_RATE, NR_OF_BINS, PRE_TRAIN_RATIO, MINI_BATCH_SIZE, \
    PROJECTION_HEAD, MASK_RATE, MSK_SCALAR, \
    NR_OF_LOOKBACK_PATCHES, NR_OF_FORECAST_PATCHES, \
    ARTIFACTS_FOLDER, NR_OF_EPOCHS, \
    NR_OF_ENCODER_BLOCKS, NR_OF_HEADS, \
    DROPOUT_RATE, ENCODER_FFN_UNITS, EMBEDDING_DIMS, \
    LEARNING_RATE, BETA_1, BETA_2

from sklearn.utils import resample

import tensorflow as tf

from tsf_model import PreProcessor, PreTraining


def get_args():
    '''
        Parses the args.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--channel',
        required=True,
        type=str,
        help='channel'
    )

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

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    '''
        Pre-trains a given channel.
        Training logs are saved in tensorboard.
        Final model is saved.
    '''
    args = get_args()

    sChannel = args.channel

    lb_train = np.load(f'{TRAINING_DATA_FOLDER}/{sChannel}/lb_train.npy')
    fc_train = np.load(f'{TRAINING_DATA_FOLDER}/{sChannel}/fc_train.npy')

    lb_train, fc_train = resample(
        lb_train,
        fc_train,
        n_samples=int(len(lb_train) * PRE_TRAIN_RATIO),
        random_state=1)

    oPreProcessor = PreProcessor(
        iPatchSize=PATCH_SIZE,
        fPatchSampleRate=PATCH_SAMPLE_RATE,
        iNrOfBins=NR_OF_BINS
    )
    dist, tre, sea = oPreProcessor.pre_process((lb_train, fc_train))

    ds_train = tf.data.Dataset.from_tensor_slices((dist, tre, sea)).batch(
        MINI_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    oModel = PreTraining(
        iNrOfEncoderBlocks=args.nr_of_encoder_blocks,
        iNrOfHeads=args.nr_of_heads,
        fDropoutRate=args.dropout_rate,
        iEncoderFfnUnits=args.encoder_ffn_units,
        iEmbeddingDims=args.embedding_dims,
        iProjectionHeadUnits=PROJECTION_HEAD,
        iPatchSize=PATCH_SIZE,
        fMskRate=MASK_RATE,
        fMskScalar=MSK_SCALAR,
        iNrOfBins=NR_OF_BINS,
        iNrOfLookbackPatches=NR_OF_LOOKBACK_PATCHES,
        iNrOfForecastPatches=NR_OF_FORECAST_PATCHES)

    oModel.compile(
        masked_autoencoder_optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            beta_1=args.beta_1,
            beta_2=args.beta_2
        ),
        contrastive_optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            beta_1=args.beta_1,
            beta_2=args.beta_2
        )
    )

    sArtifactsDirectory = f'{ARTIFACTS_FOLDER}/{sChannel}/pre_train'
    shutil.rmtree(sArtifactsDirectory, ignore_errors=True)
    os.makedirs(sArtifactsDirectory)

    class StopAtThreshold(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs={}):
            fMaeDist = logs.get('mae_dist')
            fMaeTre = logs.get('mae_tre')
            fMaeSea = logs.get('mae_sea')
            if fMaeDist <= 0.01 and fMaeTre <= 0.01 and fMaeSea <= 0.01:
                self.model.stop_training = True
                print('Stopping because threshold is achived succesfully...')

    stop_at_thershold_callback = StopAtThreshold()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f'{sArtifactsDirectory}/logs',
        histogram_freq=1
    )

    oModel.fit(
        ds_train,
        epochs=NR_OF_EPOCHS,
        verbose=2,
        callbacks=[
            tensorboard_callback,
            stop_at_thershold_callback
        ]
    )

    oModel.save(
        sArtifactsDirectory,
        overwrite=True,
        save_format='tf')

    print('Training completed.')
