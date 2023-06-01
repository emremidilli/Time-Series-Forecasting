import sys
sys.path.append( '../')
from layers.variable_selection_network import variable_selection_network

from models.pre_training import Pre_Training

from preprocessing.constants import QUANTILE_PREDICTION_DATA_FOLDER

import numpy as np

import tensorflow as tf



if __name__ == '__main__':
    iNrOfForecastPatches = 4
    iNrOfLookbackPatches = 16
    iNrOfChannels = 3


    X_dist = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_dist.npy')[:100]
    X_tre = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_tre.npy')[:100]
    X_sea = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_sea.npy')[:100]
    X_tic = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_tic.npy')[:100]
    X_known = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_known.npy')[:100]
    X_observed = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_observed.npy')[:100]
    X_static = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_static.npy')[:100]
    Y = np.load(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\Y.npy')[:100]


    oDisERT = Pre_Training()
    oTreERT = Pre_Training()
    oSeaERT = Pre_Training()
    oTicERT = Pre_Training()
    oKnoERT = Pre_Training()
    oObsERT = Pre_Training()


    oFlatten = tf.keras.layers.Flatten()
    oDense = tf.keras.layers.Dense(units = 32)

    c_s = oFlatten(X_static)
    c_s = oDense(c_s)

    c_dist = oDisERT(X_dist)
    c_tre = oTreERT(X_tre)
    c_sea = oSeaERT(X_sea)
    c_tic = oTicERT(X_tic)
    c_known = oKnoERT(X_known)
    c_observed = oObsERT(X_observed)

    def aReturnLookbackAndForecast(x):
        x = tf.reshape(x, (x.shape[0], -1, iNrOfChannels, x.shape[-1]))
        x = x[:, 1:,:,:]
        x = x[:, :-2,:,:]
        x_l = x[:, :iNrOfLookbackPatches,:,:]
        x_f = x[:, -iNrOfForecastPatches:,:,:]

        return x_l, x_f



    c_dist_l, c_dist_f = aReturnLookbackAndForecast(c_dist)
    c_tre_l, c_tre_f = aReturnLookbackAndForecast(c_tre)
    c_sea_l, c_sea_f = aReturnLookbackAndForecast(c_sea)
    c_tic_l, c_tic_f = aReturnLookbackAndForecast(c_tic)
    c_known_l, c_known_f = aReturnLookbackAndForecast(c_known)
    c_observed_l, c_observed_f = aReturnLookbackAndForecast(c_observed)


    iPatch = 5

    c_dist_t = c_dist_l[:, iPatch,: ,:]
    c_tre_t = c_tre_l[:, iPatch,: ,:]
    c_sea_t = c_sea_l[:, iPatch,: ,:]
    c_tic_t = c_tic_l[:, iPatch,: ,:]
    c_known_t = c_known_l[:, iPatch,: ,:]
    c_observed_t = c_observed_l[:, iPatch,: ,:]


    x =  tf.stack([c_dist_t, c_tre_t, c_sea_t, c_tic_t, c_known_t, c_observed_t], axis = 1)
    x = tf.reshape(x, (x.shape[0], -1, x.shape[-1]))
    x = tf.transpose(x, (0, 2,1))

    
    oVsn = variable_selection_network(
        iModelDims = 32,
        iNrOfChannels = 3 ,
        fDropout = 0.1
    )