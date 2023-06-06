import sys
sys.path.append( '../')

import keras_tuner
from hyperparameter_tuning.constants import PRE_TRAINING_CONFIG
from preprocessing.constants import HYPERPARAMETER_TUNING_FOLDER, NEXT_PATCH_PREDICTION_DATA_FOLDER, MASKED_PATCH_PREDICTION_DATA_FOLDER, SIGN_OF_PATCH_PREDICTION_DATA_FOLDER, RANK_OF_PATCH_PREDICTION_DATA_FOLDER

from training.constants import TEST_SIZE, BATCH_SIZE
from models.general_pre_training import general_pre_training

from tensorflow.data import Dataset
from tensorflow.keras.utils import split_dataset

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import AUC, MeanAbsoluteError

import numpy as np

import os

import shutil


'''
    Hp tuning model is for an encoder representation based on.
    
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
            
    Write optimum configuration to a hyperparameter config file.
'''

def get_training_test_datasets(sDatasetName):
    
    X_npp = np.load(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
    Y_npp = np.load(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')
    
    X_mpp = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
    Y_mpp = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')
    
    X_spp = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
    Y_spp = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')
    
    X_rpp = np.load(f'{RANK_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
    Y_rpp = np.load(f'{RANK_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')
    
    
    dataset = Dataset.from_tensor_slices((X_npp, Y_npp, X_mpp, Y_mpp, X_spp, Y_spp, X_rpp, Y_rpp))
    
    
    train_dataset, test_dataset = split_dataset(
        dataset,
        right_size = TEST_SIZE,
        shuffle = False
    )
    train_dataset = train_dataset.batch(BATCH_SIZE)

    
    return train_dataset, test_dataset

class hypermodel_pre_train(keras_tuner.HyperModel):
    
    def __init__(self, oLoss,oMetrics, sTaskType = None):
        super().__init__()
        self.oLoss = oLoss
        self.oMetrics = oMetrics
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
            sTaskType = self.sTaskType)
        
        

        oModel.compile(
                    loss = self.oLoss, 
                    metrics = self.oMetrics,
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
        
    
    
if __name__ == '__main__':
    
    sDatasetName = 'dist'
    train_dataset, test_dataset = get_training_test_datasets(sDatasetName)
        
    (X_npp_test, Y_npp_test, X_mpp_test, Y_mpp_test, X_spp_test, Y_spp_test, X_rpp_test, Y_rpp_test) = enumerate(test_dataset)

    for iBatchNr, (X_npp_train, Y_npp_train, X_mpp_train, Y_mpp_train, X_spp_train, Y_spp_train, X_rpp_train, Y_rpp_train) in enumerate(train_dataset):
                
        print(f'processing batch nr: {iBatchNr}')

        sRepresentationName = f'{sDatasetName.title()[:3]}ERT'
        sLogsFolder = f'{HYPERPARAMETER_TUNING_FOLDER}\\Batch_{iBatchNr}'
        if os.path.exists(sLogsFolder) == True:
            shutil.rmtree(sLogsFolder)
            
            
        tuner = keras_tuner.RandomSearch(
            hypermodel_pre_train(BinaryCrossentropy(),AUC(), 'NPP'),
            objective=keras_tuner.Objective("val_auc", direction="max"),
            max_trials=3,
            overwrite=False,
            directory=sLogsFolder,
            project_name = sRepresentationName
        )
        
        
        tuner.search(X_npp_train, Y_npp_train, epochs=2, validation_data=(X_npp_test, Y_npp_test))