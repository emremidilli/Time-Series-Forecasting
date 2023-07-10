'''
    Inputs:

        * Tokenized data.

    Analysis:

        * Mask some of the inputs tokens that are not special ones. Use MASK_RATE.

    Outputs:

        * Masked tokenized data. Output dataset is same as unmasked inputs.
'''

import numpy as np

import constants as c

import os

import shutil


MASKED_PATCH_PREDICTION_DATA_FOLDER = c.MASKED_PATCH_PREDICTION_DATA_FOLDER

TOKENIZIED_DATA_FOLDER = c.TOKENIZIED_DATA_FOLDER

MASK_RATE = c.MASK_RATE

MSK_SCALAR = c.MSK_SCALAR


FORECAST_HORIZON = c.FORECAST_HORIZON
LOOKBACK_COEFFICIENT = c.LOOKBACK_COEFFICIENT
PATCH_SIZE = c.PATCH_SIZE

iNrOfLookbackPatches = int((FORECAST_HORIZON*LOOKBACK_COEFFICIENT)/PATCH_SIZE)
iNrOfForecastPatches = int(FORECAST_HORIZON/PATCH_SIZE)


if os.path.exists(MASKED_PATCH_PREDICTION_DATA_FOLDER) == True:
    shutil.rmtree(MASKED_PATCH_PREDICTION_DATA_FOLDER)
    
    
    
aDistribution = np.load(f'{TOKENIZIED_DATA_FOLDER}\\distribution.npy')
aDynamicDigits = np.load(f'{TOKENIZIED_DATA_FOLDER}\\dynamic_digits.npy')
aTrend = np.load(f'{TOKENIZIED_DATA_FOLDER}\\trend.npy')
aSeasonality = np.load(f'{TOKENIZIED_DATA_FOLDER}\\seasonality.npy')
aKnown = np.load(f'{TOKENIZIED_DATA_FOLDER}\\known.npy')
aObserved = np.load(f'{TOKENIZIED_DATA_FOLDER}\\observed.npy')


def aGetMspDatasets(aTrueInputs):
    aMaskedInputs = aTrueInputs.copy()
    
    iNrOfSamples = aTrueInputs.shape[0]
    iNrOfFeaturesPerChannel = iNrOfForecastPatches + iNrOfLookbackPatches
    iNrOfChannels = int(aTrueInputs.shape[1]/iNrOfFeaturesPerChannel)
    
    aLookbackPatchesToMask = np.random.rand(iNrOfSamples, iNrOfLookbackPatches )
    aLookbackPatchesToMask = aLookbackPatchesToMask.argsort()[:, :int((MASK_RATE) * (iNrOfLookbackPatches))]
    aLookbackPatchesToMask.sort(axis = 1)
    
    
    aForecastPatchesToMask = np.random.rand(iNrOfSamples, iNrOfForecastPatches)
    aForecastPatchesToMask = aForecastPatchesToMask.argsort()[:, :int((MASK_RATE) * (iNrOfForecastPatches - 1))] #latest forecast doesn't have delta. 
    aForecastPatchesToMask.sort(axis = 1)

    for i in range(iNrOfChannels):

        iLookbackStartIndex = i * iNrOfFeaturesPerChannel 
        iLookbackEndIndex = iLookbackStartIndex + iNrOfLookbackPatches - 1

        iForecastStartIndex = iLookbackEndIndex+1 
        iForecastEndIndex = iForecastStartIndex + iNrOfForecastPatches - 1 

        
        for j in range(iNrOfLookbackPatches):
            ix = (aLookbackPatchesToMask == j).any(axis = 1)
            aMaskedInputs[ix, iLookbackStartIndex + j, :] = MSK_SCALAR
            
        
        for j in range(iNrOfForecastPatches):
            ix = (aForecastPatchesToMask == j).any(axis = 1)
            aMaskedInputs[ix, iForecastStartIndex + j, :] = MSK_SCALAR
            
    X = aMaskedInputs.copy()
    Y = aTrueInputs.copy()

    return X, Y
    

    
X_dist, Y_dist = aGetMspDatasets(aDistribution)
X_tic, Y_tic = aGetMspDatasets(aDynamicDigits)
X_tre, Y_tre = aGetMspDatasets(aTrend)
X_sea, Y_sea = aGetMspDatasets(aSeasonality)
X_known, Y_known = aGetMspDatasets(aKnown)
X_observed, Y_observed = aGetMspDatasets(aObserved)


os.makedirs(MASKED_PATCH_PREDICTION_DATA_FOLDER)
np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_dist.npy', X_dist)
np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_dist.npy', Y_dist)

np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_tic.npy', X_tic)
np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_tic.npy', Y_tic)

np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_tre.npy', X_tre)
np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_tre.npy', Y_tre)

np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_sea.npy', X_sea)
np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_sea.npy', Y_sea)

np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_known.npy', X_known)
np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_known.npy', Y_known)

np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_observed.npy', X_observed)
np.save(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_observed.npy', Y_observed)