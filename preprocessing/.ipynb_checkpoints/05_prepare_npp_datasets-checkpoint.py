'''
    Input:

        * Tokenized inputs.

    Analysis:

        * NSP labels should be balanced:
            * 50% true next sequence.
            * 50% false next sequence.

        * False next sequences should be built in following strategy:
            * 25% adjacent previous
            * 25% adjacent next
            * 50% shuffled patches.

    Output:

        * Tokenized data with false next sequences.
'''

import numpy as np

import constants as c

import os

import shutil


TOKENIZIED_DATA_FOLDER = c.TOKENIZIED_DATA_FOLDER

NEXT_PATCH_PREDICTION_DATA_FOLDER = c.NEXT_PATCH_PREDICTION_DATA_FOLDER

FORECAST_HORIZON = c.FORECAST_HORIZON
LOOKBACK_COEFFICIENT = c.LOOKBACK_COEFFICIENT
PATCH_SIZE = c.PATCH_SIZE

FALSE_NEXT_PATCH_RATE = c.FALSE_NEXT_PATCH_RATE

iNrOfLookbackPatches = int((FORECAST_HORIZON*LOOKBACK_COEFFICIENT)/PATCH_SIZE)
iNrOfForecastPatches = int(FORECAST_HORIZON/PATCH_SIZE)


if os.path.exists(NEXT_PATCH_PREDICTION_DATA_FOLDER) == True:
    shutil.rmtree(NEXT_PATCH_PREDICTION_DATA_FOLDER)
    
    
    
aDistribution = np.load(f'{TOKENIZIED_DATA_FOLDER}\\distribution.npy')
aDynamicDigits = np.load(f'{TOKENIZIED_DATA_FOLDER}\\dynamic_digits.npy')
aTrend = np.load(f'{TOKENIZIED_DATA_FOLDER}\\trend.npy')
aSeasonality = np.load(f'{TOKENIZIED_DATA_FOLDER}\\seasonality.npy')
aKnowns = np.load(f'{TOKENIZIED_DATA_FOLDER}\\known.npy')
aObserveds = np.load(f'{TOKENIZIED_DATA_FOLDER}\\observed.npy')



def aGetNspDatasets(aTrueInputs):
    aFalseInputs = aTrueInputs.copy()

    iNrOfTimePatches = iNrOfForecastPatches + iNrOfLookbackPatches
    iNrOfFeaturesPerChannel = iNrOfTimePatches + 4 
    iNrOfChannels = int(aTrueInputs.shape[1]/iNrOfFeaturesPerChannel)

    aFalseOutputs = np.ones(shape = (aTrueInputs.shape[0], iNrOfChannels))

    aAdjacentNext = aTrueInputs.copy()
    aAdjacentPrev = aTrueInputs.copy()
    aShuffled = aTrueInputs.copy()


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



        # slide to next
        for j in range(iForecastStartIndex, (iForecastEndIndex+1)):
            for k in range(aTrueInputs.shape[-1]):
                aAdjacentNext[:, j ,k] = np.roll(aTrueInputs[:, j ,k], iNrOfForecastPatches)

        # slide to previous
        for j in range(iForecastStartIndex, (iForecastEndIndex+1)):
                for k in range(aTrueInputs.shape[-1]):
                    aAdjacentPrev[:, j ,k] = np.roll(aTrueInputs[:, j ,k], -iNrOfForecastPatches)


        # shuffle along the patches
        aRand = np.arange(iNrOfForecastPatches)
        np.random.shuffle(aRand)

        k = 0
        for j in range(iForecastStartIndex, (iForecastEndIndex+1)):
            aShuffled[:, j ,:] = aTrueInputs[:, iForecastStartIndex:(iForecastEndIndex+1) , :][:, aRand[k],:]

            k = k + 1

        
        aRand = np.random.rand(aTrueInputs.shape[0])
        aRand2 = np.random.rand(aTrueInputs.shape[0])
        
        
        msk1 = ((aRand <= 0.50) & (aRand2 <= FALSE_NEXT_PATCH_RATE))
        msk2 = ((0.50<aRand) & (aRand<= 0.75)& (aRand2 <= FALSE_NEXT_PATCH_RATE))
        msk3 = ((0.75<=aRand) & (aRand2 <= FALSE_NEXT_PATCH_RATE))
        
        
        aFalseInputs[msk1, iForecastStartIndex:(iForecastEndIndex+1)] = aShuffled[msk1,iForecastStartIndex:(iForecastEndIndex+1)]
        aFalseInputs[msk2, iForecastStartIndex:(iForecastEndIndex+1)] = aAdjacentNext[msk2, iForecastStartIndex:(iForecastEndIndex+1)]
        aFalseInputs[msk3, iForecastStartIndex:(iForecastEndIndex+1)] = aAdjacentPrev[msk3, iForecastStartIndex:(iForecastEndIndex+1)]

        aFalseOutputs[msk1] = 0
        aFalseOutputs[msk2] = 0
        aFalseOutputs[msk3] = 0


    return aFalseInputs, aFalseOutputs



X_dist, Y_dist = aGetNspDatasets(aDistribution)
X_tic, Y_tic = aGetNspDatasets(aDynamicDigits)
X_tre, Y_tre = aGetNspDatasets(aTrend)
X_sea, Y_sea = aGetNspDatasets(aSeasonality)
X_known, Y_known = aGetNspDatasets(aKnowns)
X_observed, Y_observed = aGetNspDatasets(aObserveds)



os.makedirs(NEXT_PATCH_PREDICTION_DATA_FOLDER)
np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\X_dist.npy', X_dist)
np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\Y_dist.npy', Y_dist)

np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\X_tic.npy', X_tic)
np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\Y_tic.npy', Y_tic)

np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\X_tre.npy', X_tre)
np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\Y_tre.npy', Y_tre)

np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\X_sea.npy', X_sea)
np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\Y_sea.npy', Y_sea)

np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\X_known.npy', X_known)
np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\Y_known.npy', Y_known)

np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\X_observed.npy', X_observed)
np.save(f'{NEXT_PATCH_PREDICTION_DATA_FOLDER}\\Y_observed.npy', Y_observed)