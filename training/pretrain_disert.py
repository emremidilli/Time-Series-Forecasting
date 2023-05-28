import sys
sys.path.append( '../')

from preprocessing.constants import NEXT_PATCH_PREDICTION_DATA_FOLDER, MASKED_PATCH_PREDICTION_DATA_FOLDER, SIGN_OF_PATCH_PREDICTION_DATA_FOLDER, RANK_OF_PATCH_PREDICTION_DATA_FOLDER

from pretrain import Pretrain

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
    
    Pretrain(X_train, Y_train, 'NPP' ,sArtifactsFolder, train_c.DISERT_NPP_LR, BinaryCrossentropy(), AUC())
    
    # masked patch prediction
    X = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_dist.npy')
    Y = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_dist.npy')

    X_train = X[iTrainStart:iTrainEnd]
    Y_train = Y[iTrainStart:iTrainEnd]
    X_test = X[iTestStart:iTestEnd]
    Y_test = Y[iTestStart:iTestEnd]
    
    Pretrain(X_train, Y_train, 'MPP' ,sArtifactsFolder, train_c.DISERT_NPP_LR, MeanSquaredError(), MeanAbsoluteError())
    
    
    # sign of patch prediction
    X = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_dist.npy')
    Y = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_dist.npy')

    X_train = X[iTrainStart:iTrainEnd]
    Y_train = Y[iTrainStart:iTrainEnd]
    X_test = X[iTestStart:iTestEnd]
    Y_test = Y[iTestStart:iTestEnd]
    
    Pretrain(X_train, Y_train, 'SPP' ,sArtifactsFolder, train_c.DISERT_SPP_LR, BinaryCrossentropy(), AUC()) 
    
    
    # rank of patch prediction
    X = np.load(f'{RANK_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_dist.npy')
    Y = np.load(f'{RANK_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_dist.npy')

    X_train = X[iTrainStart:iTrainEnd]
    Y_train = Y[iTrainStart:iTrainEnd]
    X_test = X[iTestStart:iTestEnd]
    Y_test = Y[iTestStart:iTestEnd]
    
    Pretrain(X_train, Y_train, 'RPP' ,sArtifactsFolder, train_c.DISERT_SPP_LR, BinaryCrossentropy(), AUC()) 