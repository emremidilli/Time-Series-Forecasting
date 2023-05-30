import sys
sys.path.append( '../')

from preprocessing.constants import NEXT_PATCH_PREDICTION_DATA_FOLDER, MASKED_PATCH_PREDICTION_DATA_FOLDER, SIGN_OF_PATCH_PREDICTION_DATA_FOLDER, RANK_OF_PATCH_PREDICTION_DATA_FOLDER

from models.pre_training import Pre_Training

from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import AUC, MeanAbsoluteError
from tensorflow.data import Dataset
from tensorflow.keras.utils import split_dataset

import constants as train_c
import numpy as np
import os
import shutil

        
def main(sDatasetName,  fNppLr, fMppLr, fSppLr, fRppLr):
        
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
        sRepresentationName = f'{sDatasetName.title()[:3]}ERT'
        sArtifactsFolder = f'{train_c.ARTIFACTS_FOLDER}\\Batch_{iBatchNr}\\{sRepresentationName}'
        if os.path.exists(sArtifactsFolder) == True:
            shutil.rmtree(sArtifactsFolder)
        
        # next patch prediction
        oModelNpp = Pre_Training(sTaskType = 'NPP')
        oModelNpp.Train(X_npp, Y_npp, sArtifactsFolder, fNppLr, train_c.NR_OF_EPOCHS, train_c.MINI_BATCH_SIZE, BinaryCrossentropy(), AUC())

        # masked patch prediction
        oModelMpp = Pre_Training(sTaskType = 'MPP')
        oModelMpp.predict(X_mpp) #build at least once before transfer learning. (as per keras requirement)
        oModelMpp.TransferLearningForEncoder(oModelNpp)
        oModelMpp.Train(X_mpp, Y_mpp,sArtifactsFolder, fMppLr,  train_c.NR_OF_EPOCHS, train_c.MINI_BATCH_SIZE, MeanSquaredError(), MeanAbsoluteError())

        # sign of patch prediction   
        oModelSpp = Pre_Training(sTaskType = 'SPP')
        oModelSpp.predict(X_spp) #build at least once before transfer learning. (as per keras requirement)
        oModelSpp.TransferLearningForEncoder(oModelMpp)
        oModelSpp.Train(X_spp, Y_spp,sArtifactsFolder, fSppLr,  train_c.NR_OF_EPOCHS, train_c.MINI_BATCH_SIZE, BinaryCrossentropy(), AUC()) 

        # rank of patch prediction
        oModelRpp = Pre_Training(sTaskType = 'RPP')
        oModelRpp.predict(X_rpp) #build at least once before transfer learning. (as per keras requirement)
        oModelRpp.TransferLearningForEncoder(oModelSpp)
        oModelRpp.Train(X_rpp, Y_rpp,sArtifactsFolder, fRppLr,  train_c.NR_OF_EPOCHS, train_c.MINI_BATCH_SIZE,BinaryCrossentropy(), AUC()) 
    
    

if __name__ == '__main__':
    
    # main(
    #     'dist', 
    #     train_c.DISERT_NPP_LR,
    #     train_c.DISERT_MPP_LR, 
    #     train_c.DISERT_SPP_LR, 
    #     train_c.DISERT_RPP_LR
    # )

    # main(
    #     'tic',
    #     train_c.TICERT_NPP_LR,
    #     train_c.TICERT_MPP_LR, 
    #     train_c.TICERT_SPP_LR, 
    #     train_c.TICERT_RPP_LR
    # )

    # main(
    #     'tre',
    #     train_c.TREERT_NPP_LR,
    #     train_c.TREERT_MPP_LR, 
    #     train_c.TREERT_SPP_LR, 
    #     train_c.TREERT_RPP_LR
    # )

    
    main(
        'sea',
        train_c.SEAERT_NPP_LR,
        train_c.SEAERT_MPP_LR, 
        train_c.SEAERT_SPP_LR, 
        train_c.SEAERT_RPP_LR
    )
