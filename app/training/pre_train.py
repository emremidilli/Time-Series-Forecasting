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
import numpy as np

import os

import shutil

from sklearn.utils import resample

import sys

import tensorflow as tf

sys.path.append(os.path.join(sys.path[0], '..'))

from settings import TRAINING_DATA_FOLDER, PATCH_SIZE, \
    PATCH_SAMPLE_RATE, NR_OF_BINS, PRE_TRAIN_RATIO, MINI_BATCH_SIZE, \
    PROJECTION_HEAD, MASK_RATE, MSK_SCALAR, \
    NR_OF_LOOKBACK_PATCHES, NR_OF_FORECAST_PATCHES, \
    ARTIFACTS_FOLDER

from models import PreProcessor, PreTraining


if __name__ == '__main__':

    sChannel = 'EURUSD'

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
        iNrOfEncoderBlocks=2,
        iNrOfHeads=2,
        fDropoutRate=0.10,
        iEncoderFfnUnits=32,
        iEmbeddingDims=32,
        iProjectionHeadUnits=PROJECTION_HEAD,
        iPatchSize=PATCH_SIZE,
        fMskRate=MASK_RATE,
        fMskScalar=MSK_SCALAR,
        iNrOfBins=NR_OF_BINS,
        iNrOfLookbackPatches=NR_OF_LOOKBACK_PATCHES,
        iNrOfForecastPatches=NR_OF_FORECAST_PATCHES)

    oModel.compile(
        masked_autoencoder_optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5,
            beta_1=0.85
        ),
        contrastive_optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5,
            beta_1=0.85
        )
    )

    sArtifactsDirectory = f'{ARTIFACTS_FOLDER}/{sChannel}/pre_train'

    shutil.rmtree(sArtifactsDirectory, ignore_errors=True)
    os.makedirs(sArtifactsDirectory)

    # checkpoint_filepath = f'{sArtifactsDirectory}/tmp/checkpoint'
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='loss_cl',
    #     mode='min',
    #     save_best_only=True
    #     )

    # csv_logger_callback = tf.keras.callbacks.CSVLogger(
    #     f'{sArtifactsDirectory}/logs.log',
    #     separator=';',
    #     append=True
    # )

    # class StopAtThreshold(tf.keras.callbacks.Callback):
    #     def on_batch_end(self, batch, logs={}):
    #         fMaeDist = logs.get('mae_dist')
    #         fMaeTre = logs.get('mae_tre')
    #         fMaeSea = logs.get('mae_sea')
    #         if fMaeDist <= 0.05 and fMaeTre <= 0.05 and fMaeSea <= 0.05:
    #             self.model.stop_training = True
    #             print('Stopping because threshold is achived succesfully...')

    # stop_at_thershold_callback = StopAtThreshold()

    oModel.fit(
        ds_train,
        epochs=10,  # NR_OF_EPOCHS,
        verbose=2,
        # callbacks = [
        #     model_checkpoint_callback,
        #     csv_logger_callback,
        #     stop_at_thershold_callback
        # ]
    )

    print(oModel.summary())

    # oModel.save(
    #     sArtifactsDirectory,
    #     overwrite = True,
    #     save_format = 'tf'
    #     )

    # shutil.rmtree(model_checkpoint_callback, ignore_errors = True)
