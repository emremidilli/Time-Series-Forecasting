'''
    Inputs:

        * Tokenized data.

    Analysis:

        * Mask some of the inputs tokens that are not special ones. Use MASK_RATE. Signs of deltas for special tokens are set as zero.

    Outputs:

        * Signs of deltas of the tokenized data. Signs are represented in one-hot format. Output shape is (nr_of_samples, nr_of_positions, nr_of_quantiles x 3)
        * nr_of_quantiles are multiplied by 3 to represent in one-hot format.
        * 3 means {positive, negative and zero}
'''

import numpy as np

import constants as c

import os

import shutil

from sklearn.preprocessing import OneHotEncoder


SIGN_OF_PATCH_PREDICTION_DATA_FOLDER = c.SIGN_OF_PATCH_PREDICTION_DATA_FOLDER
TOKENIZIED_DATA_FOLDER = c.TOKENIZIED_DATA_FOLDER
CONSOLIDATED_CHANNEL_DATA_FOLDER = c.CONSOLIDATED_CHANNEL_DATA_FOLDER

MASK_RATE = c.MASK_RATE

MSK_SCALAR = c.MSK_SCALAR

FORECAST_HORIZON = c.FORECAST_HORIZON
LOOKBACK_COEFFICIENT = c.LOOKBACK_COEFFICIENT
PATCH_SIZE = c.PATCH_SIZE

iNrOfLookbackPatches = int((FORECAST_HORIZON*LOOKBACK_COEFFICIENT)/PATCH_SIZE)
iNrOfForecastPatches = int(FORECAST_HORIZON/PATCH_SIZE)


if os.path.exists(SIGN_OF_PATCH_PREDICTION_DATA_FOLDER) == True:
    shutil.rmtree(SIGN_OF_PATCH_PREDICTION_DATA_FOLDER)
    
    
    
aDistribution = np.load(f'{TOKENIZIED_DATA_FOLDER}\\distribution.npy')
aDynamicDigits = np.load(f'{TOKENIZIED_DATA_FOLDER}\\dynamic_digits.npy')
aTrend = np.load(f'{TOKENIZIED_DATA_FOLDER}\\trend.npy')
aSeasonality = np.load(f'{TOKENIZIED_DATA_FOLDER}\\seasonality.npy')
aKnown = np.load(f'{TOKENIZIED_DATA_FOLDER}\\known.npy')
aObserved = np.load(f'{TOKENIZIED_DATA_FOLDER}\\observed.npy')

aDeltas = np.load(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\deltas.npy')

aSignsOfDeltas = np.sign(aDeltas)


def aGetSppDatasets(aTrueInput):
    aMaskedInput = aTrueInput.copy()

    iNrOfFeaturesPerChannel = iNrOfForecastPatches + iNrOfLookbackPatches
    iNrOfChannels = int(aTrueInput.shape[1]/iNrOfFeaturesPerChannel)
    iNrOfSamples = aTrueInput.shape[0]

    aLookbackPatchesToMask = np.random.rand(iNrOfSamples, iNrOfLookbackPatches )
    aLookbackPatchesToMask = aLookbackPatchesToMask.argsort()[:, :int((MASK_RATE) * (iNrOfLookbackPatches))]
    aLookbackPatchesToMask.sort(axis = 1)


    aForecastPatchesToMask = np.random.rand(iNrOfSamples, iNrOfForecastPatches )
    aForecastPatchesToMask = aForecastPatchesToMask.argsort()[:, :int((MASK_RATE) * (iNrOfForecastPatches - 1))] #latest forecast doesn't have delta. 
    aForecastPatchesToMask.sort(axis = 1)


    aOutput = np.zeros((iNrOfSamples, aTrueInput.shape[1],aSignsOfDeltas.shape[2]))

    for i in range(iNrOfChannels):
        aDeltasOfChannel = aSignsOfDeltas[:,:,:, i]

        iLookbackStartIndex = i * iNrOfFeaturesPerChannel 
        iLookbackEndIndex = iLookbackStartIndex + iNrOfLookbackPatches - 1

        iForecastStartIndex = iLookbackEndIndex+1 
        iForecastEndIndex = iForecastStartIndex + iNrOfForecastPatches - 1 



        for j in range(iNrOfLookbackPatches):
            ix = (aLookbackPatchesToMask == j).any(axis = 1)

            aMaskedInput[ix, iLookbackStartIndex + j, :] = MSK_SCALAR

            aOutput[ix, iLookbackStartIndex + j, :] = aSignsOfDeltas[ix, j ,:, i]


            
        for j in range(iNrOfForecastPatches - 1): #1 due to latest forecast patch can not be masked.
            ix = (aForecastPatchesToMask == j).any(axis = 1)

            aMaskedInput[ix, iForecastStartIndex + j, :] = MSK_SCALAR

            aOutput[ix, iForecastStartIndex + j, :] = aSignsOfDeltas[ix, iNrOfLookbackPatches + j ,:, i]



    X = aMaskedInput.copy()
    Y = aOutput.copy()
    
    
    # do not include negative values. make sure that all the values are between 0 and 1.
    # -1 -> negative sign, 
    # 0 -> zero (neigther negative nor positive)  
    # 1 -> positive sign
    oOneHotEncoder = OneHotEncoder( categories = [[-1, 0, 1]] ,sparse=False ,handle_unknown='ignore')    
    aToReturn = np.zeros((Y.shape[0], Y.shape[1], Y.shape[2], 3)) # 3: positive, negative, zero
    for i in range(Y.shape[1]):
        for j in range(Y.shape[2]):
            arr2 = Y[:, i, [j]]
            aToReturn[:, i, j, : ] = oOneHotEncoder.fit_transform(arr2)
    
    Y = aToReturn.copy()
    
    Y = np.reshape(Y, (Y.shape[0], Y.shape[1], -1))
    
    return X, Y



X_dist, Y_dist = aGetSppDatasets(aDistribution)
X_tic, Y_tic = aGetSppDatasets(aDynamicDigits)
X_tre, Y_tre = aGetSppDatasets(aTrend)
X_sea, Y_sea = aGetSppDatasets(aSeasonality)
X_known, Y_known = aGetSppDatasets(aKnown)
X_observed, Y_observed = aGetSppDatasets(aObserved)


os.makedirs(SIGN_OF_PATCH_PREDICTION_DATA_FOLDER)
np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_dist.npy', X_dist)
np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_dist.npy', Y_dist)

np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_tic.npy', X_tic)
np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_tic.npy', Y_tic)

np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_tre.npy', X_tre)
np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_tre.npy', Y_tre)

np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_sea.npy', X_sea)
np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_sea.npy', Y_sea)

np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_known.npy', X_known)
np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_known.npy', Y_known)

np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_observed.npy', X_observed)
np.save(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_observed.npy', Y_observed)