'''
    Used to identify the hyperparameters of pre-training model.
    There are 2 phases of hyperparameter tuning:
        1st phase - tuning the architectural hyperparameters.
        2nd phase - tuning the optimizer hyperparameters.
    Lowest values of optimizer hyperparameters are used in 1st phase.
    The best hyperparameters of 1st phase is selected as
        arguments of second phase.
    Both masked patch prediction and contrastive loss are set
        as objective of hyperparameter optimization.
    Logs of tuning process are stored in a directory.
    Tensorboard can be used later on to visualize the logs.
'''
import gc

import keras_tuner as kt

import numpy as np

import os

import shutil

from sklearn.utils import resample

import sys

import tensorflow as tf

sys.path.append(os.path.join(sys.path[0], '..'))

from models.pre_training import PreTraining
from models.pre_processing import PreProcessor
from settings import TRAINING_DATA_FOLDER, PATCH_SIZE, \
    PRE_TRAIN_RATIO, NR_OF_BINS, PATCH_SAMPLE_RATE, MINI_BATCH_SIZE, \
    HYPERPARAMETER_TUNING_FOLDER, \
    OPTIMIZER_CONFIG, ARCHITECTURE_CONFIG, \
    PROJECTION_HEAD, MASK_RATE, MSK_SCALAR, \
    NR_OF_LOOKBACK_PATCHES, NR_OF_FORECAST_PATCHES


class HyperbandTuner(kt.Hyperband):
    '''
        Tunes hyperparameters by Hyperband algorithm without saving callbacks.
        Original Hyperband class is saving callbacks.
        It causes out-of-memory issue during hyperparameter search.
        In this context, Hyperband class is overriden.
        Callback saving is disregarded.
    '''

    def __init__(self, hypermodel, **kwargs):
        super().__init__(hypermodel, **kwargs)

    def run_trial(self, trial, *args, **kwargs):
        '''
            At the beginning it clears sessions and runs garbage collector
                to avoid memory issue of trainings in loop.
        '''
        tf.keras.backend.clear_session()
        gc.collect()
        tf.compat.v1.reset_default_graph()

        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        to_return = self.hypermodel.fit(hp, model, *args, **kwargs)
        del model
        return to_return


class architecture_hypermodel(kt.HyperModel):
    def __init__(self):
        super().__init__()

    def build(self, hp):
        '''
            Used to tune only the architectural hyperparameters.
            Optimizer hyperparaters are kept constant at their minimum level.
        '''

        nr_of_encoder_blocks = hp.Int(
            name='nr_of_encoder_blocks',
            min_value=ARCHITECTURE_CONFIG['nr_of_encoder_blocks'][0],
            max_value=ARCHITECTURE_CONFIG['nr_of_encoder_blocks'][1],
            step=ARCHITECTURE_CONFIG['nr_of_encoder_blocks'][2]
        )

        nr_of_heads = hp.Int(
            name='nr_of_heads',
            min_value=ARCHITECTURE_CONFIG['nr_of_heads'][0],
            max_value=ARCHITECTURE_CONFIG['nr_of_heads'][1],
            step=ARCHITECTURE_CONFIG['nr_of_heads'][2]
        )

        nr_of_ffn_units_of_encoder = hp.Int(
            name='nr_of_ffn_units_of_encoder',
            min_value=ARCHITECTURE_CONFIG['nr_of_ffn_units_of_encoder'][0],
            max_value=ARCHITECTURE_CONFIG['nr_of_ffn_units_of_encoder'][1],
            step=ARCHITECTURE_CONFIG['nr_of_ffn_units_of_encoder'][2]
        )

        embedding_dims = hp.Int(
            name='embedding_dims',
            min_value=ARCHITECTURE_CONFIG['embedding_dims'][0],
            max_value=ARCHITECTURE_CONFIG['embedding_dims'][1],
            step=ARCHITECTURE_CONFIG['embedding_dims'][2]
        )

        dropout_rate = hp.Float(
            name='dropout_rate',
            min_value=ARCHITECTURE_CONFIG['dropout_rate'][0],
            max_value=ARCHITECTURE_CONFIG['dropout_rate'][1],
            step=ARCHITECTURE_CONFIG['dropout_rate'][2]
        )

        model = PreTraining(
            iNrOfEncoderBlocks=nr_of_encoder_blocks,
            iNrOfHeads=nr_of_heads,
            fDropoutRate=dropout_rate,
            iEncoderFfnUnits=nr_of_ffn_units_of_encoder,
            iEmbeddingDims=embedding_dims,
            iProjectionHeadUnits=PROJECTION_HEAD,
            iPatchSize=PATCH_SIZE,
            fMskRate=MASK_RATE,
            fMskScalar=MSK_SCALAR,
            iNrOfBins=NR_OF_BINS,
            iNrOfLookbackPatches=NR_OF_LOOKBACK_PATCHES,
            iNrOfForecastPatches=NR_OF_FORECAST_PATCHES
        )

        model.compile(
            masked_autoencoder_optimizer=tf.keras.optimizers.Adam(
                learning_rate=OPTIMIZER_CONFIG['learning_rate'][0]
            ),
            contrastive_optimizer=tf.keras.optimizers.Adam(
                learning_rate=OPTIMIZER_CONFIG['learning_rate'][0]
            )
        )

        return model


class optimizer_hypermodel(kt.HyperModel):
    '''
        To tune only optimizer hyperparameters.
        Architectural hyperparameters are taken as arguments.
    '''

    def __init__(self, nr_of_encoder_blocks, nr_of_heads, dropout_rate,
                 nr_of_ffn_units_of_encoder, embedding_dims):
        super().__init__()

        self.nr_of_encoder_blocks = nr_of_encoder_blocks
        self.nr_of_heads = nr_of_heads
        self.dropout_rate = dropout_rate
        self.nr_of_ffn_units_of_encoder = nr_of_ffn_units_of_encoder
        self.embedding_dims = embedding_dims

    def build(self, hp):

        learning_rate = hp.Float(
            name='learning_rate',
            min_value=OPTIMIZER_CONFIG['learning_rate'][0],
            max_value=OPTIMIZER_CONFIG['learning_rate'][1],
            step=OPTIMIZER_CONFIG['learning_rate'][2]
        )

        beta_1 = hp.Float(
            name='beta_1',
            min_value=OPTIMIZER_CONFIG['beta_1'][0],
            max_value=OPTIMIZER_CONFIG['beta_1'][1],
            step=OPTIMIZER_CONFIG['beta_1'][2]
        )

        beta_2 = hp.Float(
            name='beta_2',
            min_value=OPTIMIZER_CONFIG['beta_2'][0],
            max_value=OPTIMIZER_CONFIG['beta_2'][1],
            step=OPTIMIZER_CONFIG['beta_2'][2]
        )

        model = PreTraining(
            iNrOfEncoderBlocks=self.nr_of_encoder_blocks,
            iNrOfHeads=self.nr_of_heads,
            fDropoutRate=self.dropout_rate,
            iEncoderFfnUnits=self.nr_of_ffn_units_of_encoder,
            iEmbeddingDims=self.embedding_dims,
            iProjectionHeadUnits=PROJECTION_HEAD,
            iPatchSize=PATCH_SIZE,
            fMskRate=MASK_RATE,
            fMskScalar=MSK_SCALAR,
            iNrOfBins=NR_OF_BINS,
            iNrOfLookbackPatches=NR_OF_LOOKBACK_PATCHES,
            iNrOfForecastPatches=NR_OF_FORECAST_PATCHES
        )

        model.compile(
            masked_autoencoder_optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2
            ),
            contrastive_optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2
            )
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )


if __name__ == '__main__':
    aChannels = os.listdir(TRAINING_DATA_FOLDER)

    for sChannel in aChannels:

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

        ds_train = tf.data.Dataset.from_tensor_slices(
            (dist, tre, sea)).batch(MINI_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        sLogsFolder = f'{HYPERPARAMETER_TUNING_FOLDER}/{sChannel}/pre_train'
        shutil.rmtree(sLogsFolder, ignore_errors=True)

        oTunerArchitecture = HyperbandTuner(
            architecture_hypermodel(),
            objective=[
                kt.Objective('loss_mpp', direction='min'),
                kt.Objective('loss_cl', direction='min'),
            ],
            directory=sLogsFolder,
            project_name='architecture'
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            f'{sLogsFolder}/architecture/logs')
        oTunerArchitecture.search(
            ds_train,
            # callbacks=[tensorboard_callback]
        )

        dicBestArchitecture = oTunerArchitecture.get_best_hyperparameters(
            num_trials=1)[0]

        nr_of_encoder_blocks = dicBestArchitecture.get('nr_of_encoder_blocks')
        nr_of_heads = dicBestArchitecture.get('nr_of_heads')
        dropout_rate = dicBestArchitecture.get('dropout_rate')
        nr_of_ffn_units_of_encoder = dicBestArchitecture.get(
            'nr_of_ffn_units_of_encoder')
        embedding_dims = dicBestArchitecture.get('embedding_dims')

        oTunerOptimizer = HyperbandTuner(
            optimizer_hypermodel(
                nr_of_encoder_blocks,
                nr_of_heads,
                dropout_rate,
                nr_of_ffn_units_of_encoder,
                embedding_dims),
            objective=[
                kt.Objective('loss_mpp', direction='min'),
                kt.Objective('loss_cl', direction='min'),
            ],
            directory=sLogsFolder,
            project_name='optimizer'
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            f'{sLogsFolder}/optimizer/logs')
        oTunerOptimizer.search(
            ds_train,
            # callbacks=[tensorboard_callback]
        )
