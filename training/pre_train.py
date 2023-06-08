'''

    There are 6 representations: {DisERT, TicERT, TreERT, SeaERT, KnoERT, ObsERT}
    There are 2 tasks: {NPP, MPP}
    There are 20 batches: {1...20}

    Soluton 2:
        We can pre-train only boosting models gradually.
        A model: 
            
        get batch i

        build 6 representation model
        hard train each model with NPP and MPP.

        consider each of them as a bag.

'''

import sys
sys.path.append( '../')

from preprocessing.constants import NEXT_PATCH_PREDICTION_DATA_FOLDER, MASKED_PATCH_PREDICTION_DATA_FOLDER, HYPERPARAMETER_TUNING_FOLDER, ARTIFACTS_FOLDER

from models.general_pre_training import general_pre_training

from hyperparameter_tuning.general_pre_training import oGetArchitectureTuner, oGetOptimizerTuners

from tensorflow.data import Dataset
from tensorflow.keras.utils import split_dataset

from training.constants import TEST_SIZE, BATCH_SIZE, NR_OF_EPOCHS, MINI_BATCH_SIZE, PATIENCE
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

        oTunerNpp, oTunerMpp = oGetOptimizerTuners(
                    sLogsFolder  =sOptimumHyperparametersFolder,
                    nr_of_encoder_blocks = nr_of_encoder_blocks, 
                    nr_of_heads = nr_of_heads, 
                    dropout_rate = dropout_rate, 
                    nr_of_ffn_units_of_encoder = nr_of_ffn_units_of_encoder, 
                    embedding_dims = embedding_dims
                )

        oBestNpp = oTunerNpp.get_best_hyperparameters(1)[0]
        oBestMpp = oTunerMpp.get_best_hyperparameters(1)[0]

        X_npp = np.load(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
        Y_npp = np.load(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')

        X_mpp = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
        Y_mpp = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')


        dataset = Dataset.from_tensor_slices((X_npp, Y_npp, X_mpp, Y_mpp))
        train_dataset, validation_dataset = split_dataset(
            dataset,
            right_size = TEST_SIZE,
            shuffle = False
        )
        train_dataset = train_dataset.batch(BATCH_SIZE)
        validation_dataset = validation_dataset.batch(len(validation_dataset)).get_single_element()
        X_npp_val, Y_npp_val, X_mpp_val, Y_mpp_val = validation_dataset
        

        
        for iBatchNr, (X_npp_train, Y_npp_train, X_mpp_train, Y_mpp_train) in enumerate(train_dataset):

            # delete previously created artifacts
            sArtifactsFolder = f'{ARTIFACTS_FOLDER}\\Batch_{iBatchNr}\\{sRepresentationName}'
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
            oModelNpp.predict(X_npp_train) #build at least once before transfer learning. (as per keras requirement)
            oModelNpp.Train(
                X_npp_train, 
                Y_npp_train,
                X_npp_val, 
                Y_npp_val, 
                sArtifactsFolder, 
                oBestNpp.get('learning_rate'), 
                oBestNpp.get('momentum_rate'),
                NR_OF_EPOCHS, 
                MINI_BATCH_SIZE, 
                PATIENCE
            )

            # masked patch prediction
            oModelMpp = general_pre_training(
                iNrOfEncoderBlocks = nr_of_encoder_blocks,
                iNrOfHeads = nr_of_heads,
                fDropoutRate = dropout_rate, 
                iEncoderFfnUnits = nr_of_ffn_units_of_encoder,
                iEmbeddingDims = embedding_dims, 
                sTaskType = 'MPP'
            )
            oModelMpp.predict(X_mpp_train) #build at least once before transfer learning. (as per keras requirement)
            oModelMpp.TransferLearningForEncoder(oModelNpp)
            oModelMpp.Train(
                X_mpp_train, 
                Y_mpp_train,
                X_mpp_val,
                Y_mpp_val,
                sArtifactsFolder, 
                oBestMpp.get('learning_rate'), 
                oBestMpp.get('momentum_rate'),  
                NR_OF_EPOCHS, 
                MINI_BATCH_SIZE, 
                PATIENCE
            )
            
            
            
            oFinalModel = general_pre_training(
                iNrOfEncoderBlocks = nr_of_encoder_blocks,
                iNrOfHeads = nr_of_heads,
                fDropoutRate = dropout_rate, 
                iEncoderFfnUnits = nr_of_ffn_units_of_encoder,
                iEmbeddingDims = embedding_dims, 
                sTaskType = None
            )
            oFinalModel.predict(X_mpp_train)
            oFinalModel.TransferLearningForEncoder(oModelMpp)
            oFinalModel.save( 
                filepath = f'{sArtifactsFolder}\\final_model\\', 
                overwrite=True, 
                save_format='tf'
            )