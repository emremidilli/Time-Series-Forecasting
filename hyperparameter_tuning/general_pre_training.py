import sys
sys.path.append( '../')

import keras_tuner
from hyperparameter_tuning.constants import PRE_TRAINING_CONFIG, NR_OF_EPOCHS, PATIENCE, FACTOR, SAMPLE_SIZE
from settings import HYPERPARAMETER_TUNING_FOLDER, NEXT_PATCH_PREDICTION_DATA_FOLDER, MASKED_PATCH_PREDICTION_DATA_FOLDER, SIGN_OF_PATCH_PREDICTION_DATA_FOLDER, RANK_OF_PATCH_PREDICTION_DATA_FOLDER

from settings import TEST_SIZE, MINI_BATCH_SIZE
from models.general_pre_training import general_pre_training

from tensorflow.data import Dataset
from tensorflow.keras.utils import split_dataset

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

import os

import shutil


'''
    Hp tuning model is for an encoder representation based on sampling a training dataset.
    It takes too much time to perform hyperparameter tuning based on all training dataset.
    That's why, a random sampling is done.
    
    Representation/Task pairs are searched.
    Representation = {DisERT, TicERT, TreERT, SeaERT, KnoERT, ObsERT}
    Tasks = {NPP, MPP, SPP, RPP}
    Hyperband algorithm is used.
    Hyperparameter optimization is done based on test dataset.
    
    Hyperparameters to optimize are:
        optimizer:
            fLearningRate,
            fMomentumRate,
        
        architecture:
            iNrOfEncoderBlocks,
            iNrOfHeads,
            iNrOfFfnUnitsOfEncoder,
            iEmbeddingDims,
            fDropoutRate
'''

def get_training_test_datasets(sDatasetName):
    
    X_npp = np.load(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
    Y_npp = np.load(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')
    
    X_mpp = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
    Y_mpp = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')
    

    dataset = Dataset.from_tensor_slices((X_npp, Y_npp, X_mpp, Y_mpp))
    
    train_dataset, test_dataset = split_dataset(
        dataset,
        right_size = TEST_SIZE,
        shuffle = False
    )
    
    train_dataset =  train_dataset.shuffle(buffer_size=len(train_dataset))
    train_dataset = train_dataset.take(SAMPLE_SIZE)
    train_dataset= train_dataset.batch(len(train_dataset)).get_single_element()
    test_dataset= test_dataset.batch(len(test_dataset)).get_single_element()
    
    return train_dataset, test_dataset

class architectural_hypermodel(keras_tuner.HyperModel):
        
    def build(self, hp ):
    
        nr_of_encoder_blocks = hp.Int(
            name ='nr_of_encoder_blocks',
            min_value = PRE_TRAINING_CONFIG['architecture']['nr_of_encoder_blocks'][0],
            max_value = PRE_TRAINING_CONFIG['architecture']['nr_of_encoder_blocks'][1],
            step = PRE_TRAINING_CONFIG['architecture']['nr_of_encoder_blocks'][2]
        )


        nr_of_heads = hp.Int(
            name ='nr_of_heads',
            min_value = PRE_TRAINING_CONFIG['architecture']['nr_of_heads'][0],
            max_value = PRE_TRAINING_CONFIG['architecture']['nr_of_heads'][1],
            step = PRE_TRAINING_CONFIG['architecture']['nr_of_heads'][2]
        )


        nr_of_ffn_units_of_encoder = hp.Int(
            name ='nr_of_ffn_units_of_encoder',
            min_value = PRE_TRAINING_CONFIG['architecture']['nr_of_ffn_units_of_encoder'][0],
            max_value = PRE_TRAINING_CONFIG['architecture']['nr_of_ffn_units_of_encoder'][1],
            step = PRE_TRAINING_CONFIG['architecture']['nr_of_ffn_units_of_encoder'][2]
        )


        embedding_dims = hp.Int(
            name ='embedding_dims',
            min_value = PRE_TRAINING_CONFIG['architecture']['embedding_dims'][0],
            max_value = PRE_TRAINING_CONFIG['architecture']['embedding_dims'][1],
            step = PRE_TRAINING_CONFIG['architecture']['embedding_dims'][2]
        )

        dropout_rate = hp.Float(
            name ='dropout_rate',
            min_value = PRE_TRAINING_CONFIG['architecture']['dropout_rate'][0],
            max_value = PRE_TRAINING_CONFIG['architecture']['dropout_rate'][1],
            step = PRE_TRAINING_CONFIG['architecture']['dropout_rate'][2]
        )


        oModel = general_pre_training(
            iNrOfEncoderBlocks = nr_of_encoder_blocks,
            iNrOfHeads = nr_of_heads,
            fDropoutRate = dropout_rate, 
            iEncoderFfnUnits = nr_of_ffn_units_of_encoder,
            iEmbeddingDims = embedding_dims, 
            sTaskType = 'NPP'
        )
        

        oModel.compile(
                    loss = oModel.oLoss, 
                    metrics = oModel.oMetric,
                    optimizer= Adam(
                        learning_rate=ExponentialDecay(
                            initial_learning_rate=PRE_TRAINING_CONFIG['optimizer']['learning_rate'][0],
                            decay_steps=10**2,
                            decay_rate=0.9
                        ),
                        beta_1 = PRE_TRAINING_CONFIG['optimizer']['momentum_rate'][0]
                    )
                )

        return oModel
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )
        
    
class optimizer_hypermodel(keras_tuner.HyperModel):
    
    def __init__(self, nr_of_encoder_blocks, nr_of_heads, dropout_rate, nr_of_ffn_units_of_encoder, embedding_dims, sTaskType = None):
        super().__init__()
        
        self.nr_of_encoder_blocks = nr_of_encoder_blocks
        self.nr_of_heads = nr_of_heads
        self.dropout_rate = dropout_rate
        self.nr_of_ffn_units_of_encoder = nr_of_ffn_units_of_encoder
        self.embedding_dims = embedding_dims
        self.sTaskType =sTaskType
        
    
    def build(self, hp ):
    
        learning_rate = hp.Float(
            name ='learning_rate',
            min_value = PRE_TRAINING_CONFIG['optimizer']['learning_rate'][0],
            max_value = PRE_TRAINING_CONFIG['optimizer']['learning_rate'][1],
            step = PRE_TRAINING_CONFIG['optimizer']['learning_rate'][2]
        )


        momentum_rate = hp.Float(
            name ='momentum_rate',
            min_value = PRE_TRAINING_CONFIG['optimizer']['momentum_rate'][0],
            max_value = PRE_TRAINING_CONFIG['optimizer']['momentum_rate'][1],
            step = PRE_TRAINING_CONFIG['optimizer']['momentum_rate'][2]
        )

        oModel = general_pre_training(
            iNrOfEncoderBlocks = self.nr_of_encoder_blocks,
            iNrOfHeads = self.nr_of_heads,
            fDropoutRate = self.dropout_rate, 
            iEncoderFfnUnits = self.nr_of_ffn_units_of_encoder,
            iEmbeddingDims = self.embedding_dims, 
            sTaskType = self.sTaskType
        )
        

        oModel.compile(
                    loss = oModel.oLoss, 
                    metrics = oModel.oMetric,
                    optimizer= Adam(
                        learning_rate=ExponentialDecay(
                            initial_learning_rate=learning_rate,
                            decay_steps=10**2,
                            decay_rate=0.9
                        ),
                        beta_1 = momentum_rate
                    )
                )

        return oModel
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )  
    
    
def oGetArchitectureTuner(sLogsFolder):
    oTunerArchitecture = keras_tuner.Hyperband(
        architectural_hypermodel(),
        objective=keras_tuner.Objective('val_auc', direction='max'),
        max_epochs=NR_OF_EPOCHS,
        factor = FACTOR,
        directory=f'{sLogsFolder}\\architecture',
        project_name = 'NPP'
    )
    
    return oTunerArchitecture

    
def oGetOptimizerTuners(sLogsFolder, nr_of_encoder_blocks,nr_of_heads, dropout_rate, nr_of_ffn_units_of_encoder, embedding_dims ):
        # tune optimizer hyperparamters for NPP
        oTunerNpp = keras_tuner.Hyperband(
            optimizer_hypermodel(
                nr_of_encoder_blocks = nr_of_encoder_blocks, 
                nr_of_heads = nr_of_heads, 
                dropout_rate = dropout_rate, 
                nr_of_ffn_units_of_encoder = nr_of_ffn_units_of_encoder, 
                embedding_dims = embedding_dims, 
                sTaskType = 'NPP'
            ),
            objective=keras_tuner.Objective('val_auc', direction='max'),
            max_epochs=NR_OF_EPOCHS,
            factor = FACTOR,
            directory=f'{sLogsFolder}\\optimizer',
            project_name = 'NPP'
        )

        
        # tune optimizer hyperparamters for MPP
        oTunerMpp = keras_tuner.Hyperband(
            optimizer_hypermodel(
                nr_of_encoder_blocks = nr_of_encoder_blocks, 
                nr_of_heads = nr_of_heads, 
                dropout_rate = dropout_rate, 
                nr_of_ffn_units_of_encoder = nr_of_ffn_units_of_encoder, 
                embedding_dims = embedding_dims, 
                sTaskType = 'MPP'
            ),
            objective=keras_tuner.Objective('val_mean_absolute_error', direction='min'),
            max_epochs=NR_OF_EPOCHS,
            factor = FACTOR,
            directory=f'{sLogsFolder}\\optimizer',
            project_name = 'MPP'
        )
        
        
        return oTunerNpp, oTunerMpp

    
            

if __name__ == '__main__':
    
    oEarlyStop = EarlyStopping(monitor='val_loss', patience=PATIENCE)
    
    for sDatasetName in ['dist', 'tic' ,'tre', 'sea' ,'known', 'observed']:
        print(f'processing {sDatasetName}')
        
        train_dataset, test_dataset = get_training_test_datasets(sDatasetName)
        
        X_npp_train, Y_npp_train, X_mpp_train, Y_mpp_train = train_dataset
        X_npp_test, Y_npp_test, X_mpp_test, Y_mpp_test = test_dataset
        
        sRepresentationName = f'{sDatasetName.title()[:3]}ERT'

        sLogsFolder = f'{HYPERPARAMETER_TUNING_FOLDER}\\{sRepresentationName}'
        if os.path.exists(sLogsFolder) == True:
            shutil.rmtree(sLogsFolder)

        
        # tune architecture based on SPP task

        oTunerArchitecture = oGetArchitectureTuner(sLogsFolder)
        oTunerArchitecture.search(
            X_spp_train, 
            Y_spp_train, 
            epochs=NR_OF_EPOCHS,
            batch_size = MINI_BATCH_SIZE, 
            validation_data=(X_spp_test, Y_spp_test), 
            callbacks=[oEarlyStop]
        )
        
        
        dicBestArchitecture = oTunerArchitecture.get_best_hyperparameters(num_trials=1)[0]
        
        
        nr_of_encoder_blocks = dicBestArchitecture.get('nr_of_encoder_blocks')
        nr_of_heads = dicBestArchitecture.get('nr_of_heads')
        dropout_rate = dicBestArchitecture.get('dropout_rate')
        nr_of_ffn_units_of_encoder = dicBestArchitecture.get('nr_of_ffn_units_of_encoder')
        embedding_dims = dicBestArchitecture.get('embedding_dims')
        
        
        oTunerNpp, oTunerMpp = oGetOptimizerTuners(
            sLogsFolder  =sLogsFolder,
            nr_of_encoder_blocks = nr_of_encoder_blocks, 
            nr_of_heads = nr_of_heads, 
            dropout_rate = dropout_rate, 
            nr_of_ffn_units_of_encoder = nr_of_ffn_units_of_encoder, 
            embedding_dims = embedding_dims
        )
        
        # tune optimizer hyperparamters for NPP
        oTunerNpp.search(
            X_npp_train, 
            Y_npp_train, 
            epochs=NR_OF_EPOCHS,
            batch_size = MINI_BATCH_SIZE, 
            validation_data=(X_npp_test, Y_npp_test), 
            callbacks=[oEarlyStop]
        )

        
        # tune optimizer hyperparamters for MPP
        oTunerMpp.search(
            X_mpp_train, 
            Y_mpp_train, 
            epochs=NR_OF_EPOCHS,
            batch_size = MINI_BATCH_SIZE, 
            validation_data=(X_mpp_test, Y_mpp_test), 
            callbacks=[oEarlyStop]
        )

        