import numpy as np

import constants as c

import os

import shutil


QUANTILE_PREDICTION_DATA_FOLDER = c.QUANTILE_PREDICTION_DATA_FOLDER
TOKENIZIED_DATA_FOLDER = c.TOKENIZIED_DATA_FOLDER
CONSOLIDATED_CHANNEL_DATA_FOLDER = c.CONSOLIDATED_CHANNEL_DATA_FOLDER

MSK_SCALAR = c.MSK_SCALAR

FORECAST_HORIZON = c.FORECAST_HORIZON
LOOKBACK_COEFFICIENT = c.LOOKBACK_COEFFICIENT
PATCH_SIZE = c.PATCH_SIZE

iNrOfLookbackPatches = int((FORECAST_HORIZON*LOOKBACK_COEFFICIENT)/PATCH_SIZE)
iNrOfForecastPatches = int(FORECAST_HORIZON/PATCH_SIZE)


if os.path.exists(QUANTILE_PREDICTION_DATA_FOLDER) == True:
    shutil.rmtree(QUANTILE_PREDICTION_DATA_FOLDER)
    
    
aDistribution = np.load(f'{TOKENIZIED_DATA_FOLDER}\\distribution.npy')
aDynamicDigits = np.load(f'{TOKENIZIED_DATA_FOLDER}\\dynamic_digits.npy')
aTrend = np.load(f'{TOKENIZIED_DATA_FOLDER}\\trend.npy')
aSeasonality = np.load(f'{TOKENIZIED_DATA_FOLDER}\\seasonality.npy')
aKnown = np.load(f'{TOKENIZIED_DATA_FOLDER}\\known.npy')
aObserved = np.load(f'{TOKENIZIED_DATA_FOLDER}\\observed.npy')

aQuantiles = np.load(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\quantiles.npy')

aStaticCovariates = np.load(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\static_covariates.npy')


def aGetQpDatasets(aTrueInput):
    aMaskedInput = aTrueInput.copy()

    iNrOfTimePatches = iNrOfForecastPatches + iNrOfLookbackPatches
    iNrOfFeaturesPerChannel = iNrOfTimePatches + 4 
    iNrOfChannels = int(aTrueInput.shape[1]/iNrOfFeaturesPerChannel)
    iNrOfSamples = aTrueInput.shape[0]

    for i in range(iNrOfChannels):
        # cls: beginning of each channel.
        iFirstTokenIndex = i * iNrOfFeaturesPerChannel 
        iLastTokenIndex = iFirstTokenIndex + iNrOfFeaturesPerChannel - 1 

        # lookback window: after cls 
        iLookbackStartIndex = iFirstTokenIndex+1
        iLookbackEndIndex = iLookbackStartIndex + iNrOfLookbackPatches - 1

        # forecast window: 
        iForecastStartIndex = iLookbackEndIndex+2 # (there is [SEP] between end of lookback and start of forecast)
        iForecastEndIndex = iForecastStartIndex + iNrOfForecastPatches - 1


        for j in range(iNrOfForecastPatches):
            aMaskedInput[:, iForecastStartIndex + j, :] = MSK_SCALAR

            
    X = aMaskedInput.copy() 
    
    return X



X_dist = aGetQpDatasets(aDistribution)
X_tic = aGetQpDatasets(aDynamicDigits)
X_tre = aGetQpDatasets(aTrend)
X_sea = aGetQpDatasets(aSeasonality)
X_known = aKnown.copy()
X_observed = aGetQpDatasets(aObserved)


X_static = aStaticCovariates.copy()
X_static = X_static.reshape((X_static.shape[0], 1,-1))

Y = aQuantiles.copy() # (nr_of_samples, nr_of_forecast_patches, nr_of_quantiles, nr_of_channels)

os.makedirs(QUANTILE_PREDICTION_DATA_FOLDER)
np.save(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_dist.npy', X_dist)
np.save(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_tic.npy', X_tic)
np.save(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_tre.npy', X_tre)
np.save(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_sea.npy', X_sea)
np.save(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_known.npy', X_known)
np.save(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_observed.npy', X_observed)
np.save(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\X_static.npy', X_static)
np.save(f'{QUANTILE_PREDICTION_DATA_FOLDER}\\Y.npy', Y)