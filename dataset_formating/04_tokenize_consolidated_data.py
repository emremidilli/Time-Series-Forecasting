'''
    Input:

        * Consolidated numpy datasets

    Output:

        * Tokenized data for pre-training purpose. Format:[Lookback Tokens][Forecast Tokens]...
'''


import numpy as np

import constants as c

import os

import shutil


CONSOLIDATED_CHANNEL_DATA_FOLDER = c.CONSOLIDATED_CHANNEL_DATA_FOLDER
TOKENIZIED_DATA_FOLDER = c.TOKENIZIED_DATA_FOLDER


FORECAST_HORIZON = c.FORECAST_HORIZON
LOOKBACK_COEFFICIENT = c.LOOKBACK_COEFFICIENT
PATCH_SIZE = c.PATCH_SIZE


iNrOfLookbackPatches = int((FORECAST_HORIZON*LOOKBACK_COEFFICIENT)/PATCH_SIZE)
iNrOfForecastPatches = int(FORECAST_HORIZON/PATCH_SIZE)



aDistribution = np.load(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\distribution.npy')
aDynamicDigits = np.load(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\dynamic_digits.npy')
aTrend = np.load(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\trend.npy')
aSeasonality = np.load(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\seasonality.npy')
aKnown = np.load(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\knowns.npy')
aObserved = np.load(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\observeds.npy')


if os.path.exists(TOKENIZIED_DATA_FOLDER) == True:
    shutil.rmtree(TOKENIZIED_DATA_FOLDER)
    
def aTokenize(arr):

    iNrOfChannels = arr.shape[-1]
    iNrOfTimePatches = arr.shape[1]
    iNrOfFeatures=  arr.shape[2]
    iNrOfSamples = arr.shape[0]

    iNrOfFeaturesPerChannel = (iNrOfTimePatches) 
    iNrOfPositions = iNrOfChannels * iNrOfFeaturesPerChannel  

    aToReturn = np.zeros(shape = (iNrOfSamples, iNrOfPositions,iNrOfFeatures )) - 1


    for i in range(iNrOfChannels):

        iLookbackStartIndex = i * iNrOfFeaturesPerChannel 
        iLookbackEndIndex = iLookbackStartIndex + iNrOfLookbackPatches - 1
        
        
        
        aToReturn[:, iLookbackStartIndex:(iLookbackEndIndex+1) ,:] = arr[:, 0:iNrOfLookbackPatches, :, i]


        iForecastStartIndex = iLookbackEndIndex+1
        iForecastEndIndex = iForecastStartIndex + iNrOfForecastPatches - 1 
        

        aToReturn[:, iForecastStartIndex:(iForecastEndIndex+1) ,:] = arr[:, -iNrOfForecastPatches:, :, i]


    return aToReturn


aDistribution = aTokenize(aDistribution)
aDynamicDigits = aTokenize(aDynamicDigits)
aTrend = aTokenize(aTrend)
aSeasonality = aTokenize(aSeasonality)
aKnown = aTokenize(aKnown)
aObserved = aTokenize(aObserved)


os.makedirs(TOKENIZIED_DATA_FOLDER)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\distribution.npy', aDistribution)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\dynamic_digits.npy', aDynamicDigits)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\trend.npy', aTrend)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\seasonality.npy', aSeasonality)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\known.npy', aKnown)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\observed.npy', aObserved)