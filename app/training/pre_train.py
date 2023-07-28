'''
    Trains a pre-training model for a univariate forecasting model.
    loads training datasets
    builds a pre-training model
    trains and saves pre-training logs

    inputs:
        lb_train: (None, timesteps)
        fc_train: (None, timesteps)
'''


import sys

import os
import shutil

# sys.path.append( '../')
sys.path.append(os.path.join(sys.path[0], '..'))

from settings import *

from models import *

import numpy as np

from keras.optimizers import Adam


if __name__ == '__main__':


    sChannel = 'EURUSD'

    lb_train = np.load(f'{TRAINING_DATA_FOLDER}/{sChannel}/lb_train.npy')[: 160]
    fc_train = np.load(f'{TRAINING_DATA_FOLDER}/{sChannel}/fc_train.npy')[: 160]

    ds_lb_train = tf.data.Dataset.from_tensor_slices(lb_train)
    ds_fc_train = tf.data.Dataset.from_tensor_slices(fc_train)

    oPreProcessor = PreProcessor(
        iPatchSize = PATCH_SIZE,
        fPatchSampleRate = PATCH_SAMPLE_RATE,
        iNrOfBins = NR_OF_BINS
    )
    dist, tre,shape = oPreProcessor((ds_lb_train,ds_fc_train))
    print(dist)

    '''
    oModel = PreTraining(
                 iNrOfEncoderBlocks = 2,
                 iNrOfHeads = 2,
                 fDropoutRate = 0.10,
                 iEncoderFfnUnits = 32,
                 iEmbeddingDims = 32,
                 iProjectionHeadUnits = 32,
                 iPatchSize = PATCH_SIZE,
                 fMskRate = MASK_RATE,
                 fMskScalar = MSK_SCALAR,
                 iNrOfBins = NR_OF_BINS,
                 iNrOfLookbackPatches= NR_OF_LOOKBACK_PATCHES,
                 iNrOfForecastPatches= NR_OF_FORECAST_PATCHES
    )


    oModel.compile(
        masked_autoencoder_optimizer= Adam(
            learning_rate=1e-5,
            beta_1 = 0.85
        ),
        contrastive_optimizer= Adam(
            learning_rate=1e-5,
            beta_1 = 0.85
        )
    )

    lb_dist = tf.cast(lb_dist, tf.float32)
    lb_tre = tf.cast(lb_tre, tf.float32)
    lb_sea = tf.cast(lb_sea, tf.float32)
    fc_dist = tf.cast(fc_dist, tf.float32)
    fc_tre = tf.cast(fc_tre, tf.float32)
    fc_sea = tf.cast(fc_sea, tf.float32)


    sArtifactsDirectory = f'{ARTIFACTS_FOLDER}/{sChannel}/pre_train'

    shutil.rmtree(sArtifactsDirectory, ignore_errors = True)
    os.makedirs(sArtifactsDirectory)

    checkpoint_filepath = f'{sArtifactsDirectory}/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss_cl',
        mode='min',
        save_best_only=True
        )

    csv_logger_callback = tf.keras.callbacks.CSVLogger(
        f'{sArtifactsDirectory}/logs.log',
        separator=';',
        append=True
    )

    class StopAtThreshold(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs={}):
            if logs.get('mae_dist') <= 0.05 and logs.get('mae_tre') <= 0.05 and logs.get('mae_sea') <= 0.05 :
                self.model.stop_training = True
                print('Stopping because threshold is achived succesfully...')

    stop_at_thershold_callback = StopAtThreshold()

    oModel.fit(
        (dist, tre, sea),
        (dist, tre, sea),
        epochs= 5, #NR_OF_EPOCHS
        batch_size=MINI_BATCH_SIZE,
        verbose=1,
        callbacks = [
            model_checkpoint_callback,
            csv_logger_callback,
            stop_at_thershold_callback
        ]
    )


    oModel.save(
        sArtifactsDirectory,
        overwrite = True,
        save_format = 'tf'
        )

    shutil.rmtree(model_checkpoint_callback, ignore_errors = True)
    '''

