import sys
sys.path.append( '../')
from models.temporal_fusion_transformer import temporal_fusion_transformer

from models.general_pre_training import general_pre_training

from settings import QUANTILE_PREDICTION_DATA_FOLDER
from settings import TARGET_QUANTILES

import numpy as np

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.utils import split_dataset


from settings import *

from tensorflow.keras.metrics import MeanAbsoluteError

import os

import shutil


iNrOfForecastPatches = 4
iNrOfLookbackPatches = 16
iNrOfChannels = 3

# split lookback and forecast windows from tokenized version
def aReturnLookbackAndForecast(x):
    x = tf.reshape(x, (x.shape[0], -1, iNrOfChannels, x.shape[-1]))
    x = x[:, 1:,:,:]
    x = x[:, :-2,:,:]
    x_l = x[:, :iNrOfLookbackPatches,:,:]
    x_f = x[:, -iNrOfForecastPatches:,:,:]

    return x_l, x_f

if __name__ == '__main__':
    # read input data
    X_dist = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_dist.npy')
    X_tre = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_tre.npy')
    X_sea = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_sea.npy')
    X_tic = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_tic.npy')
    X_known = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_known.npy')
    X_observed = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_observed.npy')
    X_static = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_static.npy')
    Y = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\Y.npy')
    
    
    # convert to dataset
    dataset = Dataset.from_tensor_slices((X_dist, X_tre, X_sea, X_tic, X_known, X_observed, X_static, Y))
    train_dataset, _ = split_dataset(
        dataset,
        right_size = TEST_SIZE,
        shuffle = False
    )
    train_dataset = train_dataset.batch(BATCH_SIZE)
    
    # process with pre-trained models
    oDisERT = general_pre_training()
    oTreERT = general_pre_training()
    oSeaERT = general_pre_training()
    oTicERT = general_pre_training()
    oKnoERT = general_pre_training()
    oObsERT = general_pre_training()
    
    for iBatchNr, (X_dist, X_tre, X_sea, X_tic, X_known, X_observed, X_static, Y) in enumerate(train_dataset):
        
        c_dist = oDisERT(X_dist)
        c_tre = oTreERT(X_tre)
        c_sea = oSeaERT(X_sea)
        c_tic = oTicERT(X_tic)
        c_known = oKnoERT(X_known)
        c_observed = oObsERT(X_observed)

        
        c_dist_l, c_dist_f = aReturnLookbackAndForecast(c_dist)
        c_tre_l, c_tre_f = aReturnLookbackAndForecast(c_tre)
        c_sea_l, c_sea_f = aReturnLookbackAndForecast(c_sea)
        c_tic_l, c_tic_f = aReturnLookbackAndForecast(c_tic)
        c_known_l, c_known_f = aReturnLookbackAndForecast(c_known)
        c_observed_l, c_observed_f = aReturnLookbackAndForecast(c_observed)

        x_l =  tf.stack([c_dist_l, c_tre_l, c_sea_l, c_tic_l, c_known_l, c_observed_l], axis = 2)
        x_l = tf.reshape(x_l, (x_l.shape[0],x_l.shape[1],-1, x_l.shape[-1]))
        x_l = tf.transpose(x_l, (0,1, 3,2))

        x_f =  tf.stack([c_dist_f, c_tre_f, c_sea_f, c_tic_f, c_known_f, c_observed_f], axis = 2)
        x_f = tf.reshape(x_f, (x_f.shape[0],x_f.shape[1],-1, x_f.shape[-1]))
        x_f = tf.transpose(x_f, (0,1, 3,2))
        

        sArtifactsFolder = f'{ARTIFACTS_FOLDER}\\Batch_{iBatchNr}\\TFT'
        if os.path.exists(sArtifactsFolder) == True:
            shutil.rmtree(sArtifactsFolder)

        oTft = temporal_fusion_transformer(
            iNrOfLookbackPatches, 
            iNrOfForecastPatches,
            TARGET_QUANTILES,
            fDropout = 0.1,
            iModelDims = 32,
            iNrOfChannels = 3
        )
        oTft.Train(
            X_train = (x_l, x_f, X_static), 
            Y_train = Y,
            sArtifactsFolder = sArtifactsFolder,
            fLearningRate = 0.01,
            iNrOfEpochs =  NR_OF_EPOCHS, 
            iBatchSize = MINI_BATCH_SIZE,
            oLoss = oTft.quantile_loss,
            oMetrics = MeanAbsoluteError()
        )