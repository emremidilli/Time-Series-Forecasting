import sys
sys.path.append( '../')

from training.constants import TEST_SIZE, NR_OF_EPOCHS, MINI_BATCH_SIZE, BATCH_SIZE

from hyperparameter_tuning.general_pre_training import oGetArchitectureTuner

from models.bootstrap_aggregation import bootstrap_aggregation
from models.general_pre_training import general_pre_training

import numpy as np

import os

from preprocessing.constants import SIGN_OF_PATCH_PREDICTION_DATA_FOLDER, RANK_OF_PATCH_PREDICTION_DATA_FOLDER, HYPERPARAMETER_TUNING_FOLDER, ARTIFACTS_FOLDER

from tensorflow.data import Dataset
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import split_dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC

import shutil


def aGetRepresentationModels(sRepresentationName):
    aModels = []
    for root, dirs, files in os.walk(ARTIFACTS_FOLDER):
        for sFolderName in dirs:
            
            if sFolderName == sRepresentationName:
                sFoundFolderPath = (os.path.join(root, sFolderName))
                
                sBatchName = os.path.basename(root)
                
                if 'final_model' in os.listdir(sFoundFolderPath):
                    
                    oModel = load_model(
                        filepath = f'{sFoundFolderPath}\\final_model'
                    )
                    
                    aModels.append( oModel)
                    
                    
                   
    return aModels
                    

if __name__ == '__main__':
     
    for sDatasetName in ['dist', 'tic', 'tre','sea', 'known', 'observed', ]:
        sRepresentationName = f'{sDatasetName.title()[:3]}ERT'
        sModelArtifactPath = f'{ARTIFACTS_FOLDER}\\Aggregation\\{sRepresentationName}\\'
        if os.path.exists(sModelArtifactPath) == True:
            shutil.rmtree(sModelArtifactPath)

        os.makedirs(sModelArtifactPath)           


        X_spp = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
        Y_spp = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')

        Y_rpp = np.load(f'{RANK_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')


        dataset = Dataset.from_tensor_slices((X_spp, Y_spp, Y_rpp))
        train_dataset, validation_dataset = split_dataset(
            dataset,
            right_size = TEST_SIZE,
            shuffle = False
        )

        train_dataset = train_dataset.shuffle(len(train_dataset)).take(BATCH_SIZE)
        train_dataset = train_dataset.batch(len(train_dataset)).get_single_element()
        validation_dataset = validation_dataset.batch(len(validation_dataset)).get_single_element()


        X_spp_train, Y_spp_train, Y_rpp_train = train_dataset
        X_spp_val, Y_spp_val, Y_rpp_val = validation_dataset


        aRepresentationModels = aGetRepresentationModels(sRepresentationName)

        # set representation model layers not trainable
        for oModel in aRepresentationModels:
            for oLayer in oModel.layers:
                oLayer.trainable = False


        oAggregationModel = bootstrap_aggregation(aRepresentationModels)

        oCsvLogger = CSVLogger(f'{sModelArtifactPath}logs.log', separator=";", append=False)



        oReduceLr = ReduceLROnPlateau(
            monitor='loss', 
            factor=0.2, 
            patience= 3, 
            min_lr=0.0001
        )

        oEarlyStopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )


        oAggregationModel.compile(
            loss = BinaryCrossentropy(), 
            metrics = AUC(name = 'AUC'),
            optimizer= Adam(
                learning_rate=ExponentialDecay(
                    initial_learning_rate=1e-3,
                    decay_steps=100000,
                    decay_rate=0.96
                ),
                beta_1 = 0.90
            )
        )


        oAggregationModel.fit(
            X_spp_train, 
            [Y_spp_train, Y_rpp_train], 
            epochs= NR_OF_EPOCHS, 
            batch_size= MINI_BATCH_SIZE, 
            verbose=1,
            validation_data = (X_spp_val, [Y_spp_val, Y_rpp_val]),
            validation_batch_size = MINI_BATCH_SIZE,
            callbacks = [oCsvLogger, oReduceLr, oEarlyStopping]
        )

        oAggregationModel.save_weights(
            sModelArtifactPath,
            save_format ='tf'
        )


        oFinalModel = bootstrap_aggregation(aRepresentationModels, b_with_decoder = False)
        oFinalModel.predict(X_spp_train) # needed for just transfoer learning.
        oFinalModel.TransferLearning(oAggregationModel)
        oFinalModel.save( 
            filepath = f'{sModelArtifactPath}\\final_model\\', 
            overwrite=True, 
            save_format='tf'
        )