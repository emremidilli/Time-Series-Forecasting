import sys
sys.path.append( '../')

from preprocessing.constants import NEXT_PATCH_PREDICTION_DATA_FOLDER, MASKED_PATCH_PREDICTION_DATA_FOLDER, SIGN_OF_PATCH_PREDICTION_DATA_FOLDER, RANK_OF_PATCH_PREDICTION_DATA_FOLDER, HYPERPARAMETER_TUNING_FOLDER

from models.general_pre_training import general_pre_training

from hyperparameter_tuning.pre_train import oGetArchitectureTuner, oGetOptimizerTuners

from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import AUC, MeanAbsoluteError
from tensorflow.data import Dataset
from tensorflow.keras.utils import split_dataset

import constants as train_c
import numpy as np
import os
import shutil


if __name__ == '__main__':
    
    for sDatasetName in ['dist', 'tic', 'tre', 'sea', 'known', 'observed']:
    
        sRepresentationName = f'{sDatasetName.title()[:3]}ERT'

        sOptimumHyperparametersFolder = f'{HYPERPARAMETER_TUNING_FOLDER}\\{sRepresentationName}'

        oTunerArchitecture = oGetArchitectureTuner(
            sLogsFolder = sOptimumHyperparametersFolder
        )

        oBestArchitecture = oTunerArchitecture.get_best_hyperparameters(1)[0]

        nr_of_encoder_blocks = oBestArchitecture.get('nr_of_encoder_blocks')
        nr_of_heads = oBestArchitecture.get('nr_of_heads')
        dropout_rate = oBestArchitecture.get('dropout_rate')
        nr_of_ffn_units_of_encoder = oBestArchitecture.get('nr_of_ffn_units_of_encoder')
        embedding_dims = oBestArchitecture.get('embedding_dims')

        oTunerNpp, oTunerMpp, oTunerSpp, oTunerRpp = oGetOptimizerTuners(
                    sLogsFolder  =sOptimumHyperparametersFolder,
                    nr_of_encoder_blocks = nr_of_encoder_blocks, 
                    nr_of_heads = nr_of_heads, 
                    dropout_rate = dropout_rate, 
                    nr_of_ffn_units_of_encoder = nr_of_ffn_units_of_encoder, 
                    embedding_dims = embedding_dims
                )

        oBestNpp = oTunerNpp.get_best_hyperparameters(1)[0]
        oBestMpp = oTunerMpp.get_best_hyperparameters(1)[0]
        oBestSpp = oTunerSpp.get_best_hyperparameters(1)[0]
        oBestRpp = oTunerRpp.get_best_hyperparameters(1)[0]
        

        X_npp = np.load(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
        Y_npp = np.load(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')

        X_mpp = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
        Y_mpp = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')

        X_spp = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
        Y_spp = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')

        X_rpp = np.load(f'{RANK_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
        Y_rpp = np.load(f'{RANK_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')


        dataset = Dataset.from_tensor_slices((X_npp, Y_npp, X_mpp, Y_mpp, X_spp, Y_spp, X_rpp, Y_rpp))
        train_dataset, _ = split_dataset(
            dataset,
            right_size = train_c.TEST_SIZE,
            shuffle = False
        )
        train_dataset = train_dataset.batch(train_c.BATCH_SIZE)


        for iBatchNr, (X_npp, Y_npp, X_mpp, Y_mpp, X_spp, Y_spp, X_rpp, Y_rpp) in enumerate(train_dataset):

            # delete previously created artifacts
            sArtifactsFolder = f'{train_c.ARTIFACTS_FOLDER}\\Batch_{iBatchNr}\\{sRepresentationName}'
            if os.path.exists(sArtifactsFolder) == True:
                shutil.rmtree(sArtifactsFolder)

            # next patch prediction
            oModelNpp = general_pre_training(
                iNrOfEncoderBlocks = nr_of_encoder_blocks,
                iNrOfHeads = nr_of_heads,
                fDropoutRate = dropout_rate, 
                iEncoderFfnUnits = nr_of_ffn_units_of_encoder,
                iEmbeddingDims = embedding_dims, 
                sTaskType = 'NPP'
            )
            oModelNpp.Train(X_npp, Y_npp, sArtifactsFolder, oBestNpp.get('learning_rate'), oBestNpp.get('momentum_rate') ,train_c.NR_OF_EPOCHS, train_c.MINI_BATCH_SIZE)

            # masked patch prediction
            oModelMpp = general_pre_training(
                iNrOfEncoderBlocks = nr_of_encoder_blocks,
                iNrOfHeads = nr_of_heads,
                fDropoutRate = dropout_rate, 
                iEncoderFfnUnits = nr_of_ffn_units_of_encoder,
                iEmbeddingDims = embedding_dims, 
                sTaskType = 'MPP'
            )
            oModelMpp.predict(X_mpp) #build at least once before transfer learning. (as per keras requirement)
            oModelMpp.TransferLearningForEncoder(oModelNpp)
            oModelMpp.Train(X_mpp, Y_mpp,sArtifactsFolder, oBestMpp.get('learning_rate'), oBestMpp.get('momentum_rate'),  train_c.NR_OF_EPOCHS, train_c.MINI_BATCH_SIZE)

            # sign of patch prediction   
            oModelSpp = general_pre_training(
                iNrOfEncoderBlocks = nr_of_encoder_blocks,
                iNrOfHeads = nr_of_heads,
                fDropoutRate = dropout_rate, 
                iEncoderFfnUnits = nr_of_ffn_units_of_encoder,
                iEmbeddingDims = embedding_dims, 
                sTaskType = 'SPP'
            )
            oModelSpp.predict(X_spp) #build at least once before transfer learning. (as per keras requirement)
            oModelSpp.TransferLearningForEncoder(oModelMpp)
            oModelSpp.Train(X_spp, Y_spp,sArtifactsFolder,  oBestSpp.get('learning_rate'), oBestSpp.get('momentum_rate'),  train_c.NR_OF_EPOCHS, train_c.MINI_BATCH_SIZE) 

            # rank of patch prediction
            oModelRpp = general_pre_training(
                iNrOfEncoderBlocks = nr_of_encoder_blocks,
                iNrOfHeads = nr_of_heads,
                fDropoutRate = dropout_rate, 
                iEncoderFfnUnits = nr_of_ffn_units_of_encoder,
                iEmbeddingDims = embedding_dims, 
                sTaskType = 'RPP'
            )
            oModelRpp.predict(X_rpp) #build at least once before transfer learning. (as per keras requirement)
            oModelRpp.TransferLearningForEncoder(oModelSpp)
            oModelRpp.Train(X_rpp, Y_rpp,sArtifactsFolder,  oBestRpp.get('learning_rate'), oBestRpp.get('momentum_rate'),  train_c.NR_OF_EPOCHS, train_c.MINI_BATCH_SIZE) 