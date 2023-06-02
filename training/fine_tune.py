import sys
sys.path.append( '../')
from models.temporal_fusion_transformer import temporal_fusion_transformer

from models.general_pre_training import general_pre_training

from preprocessing.constants import QUANTILE_PREDICTION_DATA_FOLDER

import numpy as np

import tensorflow as tf


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
    X_dist = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_dist.npy')[:100]
    X_tre = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_tre.npy')[:100]
    X_sea = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_sea.npy')[:100]
    X_tic = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_tic.npy')[:100]
    X_known = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_known.npy')[:100]
    X_observed = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_observed.npy')[:100]
    X_static = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_static.npy')[:100]
    Y = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\Y.npy')[:100]
    
    
    # convert to dataset
    
    
    # process with pre-trained models
    oDisERT = general_pre_training()
    oTreERT = general_pre_training()
    oSeaERT = general_pre_training()
    oTicERT = general_pre_training()
    oKnoERT = general_pre_training()
    oObsERT = general_pre_training()

    
    
    # static covariate encoder (should be seperately developed)
    oFlatten = tf.keras.layers.Flatten()
    oDense = tf.keras.layers.Dense(units = 32)
    oLookbackRepeat = tf.keras.layers.RepeatVector(n = iNrOfLookbackPatches)
    oForecastRepeat = tf.keras.layers.RepeatVector(n = iNrOfForecastPatches)
    
    c_s = oFlatten(X_static)
    c_s = oDense(c_s)
    c_s_l = oLookbackRepeat(c_s)
    c_s_f = oForecastRepeat(c_s)
    
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
    
    
    oTft = temporal_fusion_transformer()
    print(oTft([x_l, c_s_l, x_f,c_s_f]).shape)
    
    print(oTft.summary())
    
    
    
    