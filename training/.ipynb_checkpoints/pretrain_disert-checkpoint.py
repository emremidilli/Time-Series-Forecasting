import sys
sys.path.append( '../')

from preprocessing.constants import NEXT_PATCH_PREDICTION_DATA_FOLDER, MASKED_PATCH_PREDICTION_DATA_FOLDER, SIGN_OF_PATCH_PREDICTION_DATA_FOLDER, RANK_OF_PATCH_PREDICTION_DATA_FOLDER

from models.pre_training import Pre_Training

from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import AUC, MeanAbsoluteError

import constants as train_c
import numpy as np
import os
import shutil

if __name__ == '__main__':
    
    iFoldNr = 1
    iModelNr = 1
    
    # delete previously created artifacts
    sArtifactsFolder = f'{train_c.ARTIFACTS_FOLDER}\\Fold_{iFoldNr}\\DisERT_{iModelNr}'
    if os.path.exists(sArtifactsFolder) == True:
        shutil.rmtree(sArtifactsFolder)
        
    iTrainStart = (iFoldNr-1) * train_c.FOLD_SIZE
    iTrainEnd = (iFoldNr) * train_c.FOLD_SIZE
    iTestStart = iTrainEnd
    iTestEnd = (iFoldNr + 1) * train_c.FOLD_SIZE
    
    # next patch prediction
    X = np.load(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\X_dist.npy')
    Y = np.load(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\Y_dist.npy')

    X_train = X[iTrainStart:iTrainEnd]
    Y_train = Y[iTrainStart:iTrainEnd]
    X_test = X[iTestStart:iTestEnd]
    Y_test = Y[iTestStart:iTestEnd]
    
    oModelNpp = Pre_Training(sTaskType = 'NPP')
    oModelNpp.Train(X_train, Y_train, sArtifactsFolder, train_c.DISERT_NPP_LR, train_c.NR_OF_EPOCHS, train_c.BATCH_SIZE, BinaryCrossentropy(), AUC())
        
    
    # masked patch prediction
    X = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_dist.npy')
    Y = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_dist.npy')

    X_train = X[iTrainStart:iTrainEnd]
    Y_train = Y[iTrainStart:iTrainEnd]
    X_test = X[iTestStart:iTestEnd]
    Y_test = Y[iTestStart:iTestEnd]
    
    oModelMpp = Pre_Training(sTaskType = 'MPP')
    oModelMpp.predict(X_train[[0]]) #build at least once before transfer learning. (as per keras requirement)
    oModelMpp.TransferLearningForEncoder(oModelNpp)
    oModelMpp.Train(X_train, Y_train,sArtifactsFolder, train_c.DISERT_MPP_LR,  train_c.NR_OF_EPOCHS, train_c.BATCH_SIZE, MeanSquaredError(), MeanAbsoluteError())
    
    
    # sign of patch prediction
    X = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_dist.npy')
    Y = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_dist.npy')

    X_train = X[iTrainStart:iTrainEnd]
    Y_train = Y[iTrainStart:iTrainEnd]
    X_test = X[iTestStart:iTestEnd]
    Y_test = Y[iTestStart:iTestEnd]
    
    
    oModelSpp = Pre_Training(sTaskType = 'SPP')
    oModelSpp.predict(X_train[[0]]) #build at least once before transfer learning. (as per keras requirement)
    oModelSpp.TransferLearningForEncoder(oModelMpp)
    oModelSpp.Train(X_train, Y_train,sArtifactsFolder, train_c.DISERT_SPP_LR,  train_c.NR_OF_EPOCHS, train_c.BATCH_SIZE, BinaryCrossentropy(), AUC()) 
    
    
    # rank of patch prediction
    X = np.load(f'{RANK_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_dist.npy')
    Y = np.load(f'{RANK_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_dist.npy')

    X_train = X[iTrainStart:iTrainEnd]
    Y_train = Y[iTrainStart:iTrainEnd]
    X_test = X[iTestStart:iTestEnd]
    Y_test = Y[iTestStart:iTestEnd]
    
    oModelRpp = Pre_Training(sTaskType = 'RPP')
    oModelRpp.predict(X_train[[0]]) #build at least once before transfer learning. (as per keras requirement)
    oModelRpp.TransferLearningForEncoder(oModelSpp)
    oModelRpp.Train(X_train, Y_train,sArtifactsFolder, train_c.DISERT_RPP_LR,  train_c.NR_OF_EPOCHS, train_c.BATCH_SIZE,BinaryCrossentropy(), AUC()) 