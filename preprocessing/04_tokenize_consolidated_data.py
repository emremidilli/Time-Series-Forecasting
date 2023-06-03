'''
    Input:

        * Consolidated numpy datasets

    Output:

        * Tokenized data for pre-training purpose. Format:[CLS][Lookback Tokens][SEP][Forecast Tokens][SEP][CNL]...
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


CLS_SCALAR = c.CLS_SCALAR
CNL_SCALAR = c.CNL_SCALAR
SEP_SCALAR  = c.SEP_SCALAR

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
    
    
# add [CLS],[CNL] and [SEP] tokens
def aAddSpecialTokens(arr):

    iNrOfChannels = arr.shape[-1]
    iNrOfTimePatches = arr.shape[1]
    iNrOfFeatures=  arr.shape[2]
    iNrOfSamples = arr.shape[0]

    iNrOfFeaturesPerChannel = (iNrOfTimePatches  + 4) # each channel has tokens 1x[CLS], 1x[CNL] and 2x[SEP] (one for end of lookback, one for end of forecast horizon)
    iNrOfPositions = iNrOfChannels * iNrOfFeaturesPerChannel  

    aToReturn = np.zeros(shape = (iNrOfSamples, iNrOfPositions,iNrOfFeatures )) - 1


    for i in range(iNrOfChannels):

        iFirstTokenIndex = i * iNrOfFeaturesPerChannel 
        iLastTokenIndex = iFirstTokenIndex + iNrOfFeaturesPerChannel - 1 


        # cls: beginning of each channel.
        aToReturn[:, iFirstTokenIndex ,:] = CLS_SCALAR

        # cnl: end of each channel 
        aToReturn[:, iLastTokenIndex ,:] = CNL_SCALAR

        # lookback window: after cls 
        iLookbackStartIndex = iFirstTokenIndex+1
        iLookbackEndIndex = iLookbackStartIndex + iNrOfLookbackPatches - 1

        aToReturn[:, iLookbackStartIndex:(iLookbackEndIndex+1) ,:] = arr[:, 0:iNrOfLookbackPatches, :, i]


        # forecast window: 
        iForecastStartIndex = iLookbackEndIndex+2 # (there is [SEP] between end of lookback and start of forecast)
        iForecastEndIndex = iForecastStartIndex + iNrOfForecastPatches - 1 

        aToReturn[:, iForecastStartIndex:(iForecastEndIndex+1) ,:] = arr[:, -iNrOfForecastPatches:, :, i]


        # sep
        aToReturn[:, iLookbackEndIndex + 1 ,:] = SEP_SCALAR # first [sep] is after end of lookback
        aToReturn[:, iForecastEndIndex + 1 ,:] = SEP_SCALAR # second [sep] is after end of foreacast


    return aToReturn


aDistribution = aAddSpecialTokens(aDistribution)
aDynamicDigits = aAddSpecialTokens(aDynamicDigits)
aTrend = aAddSpecialTokens(aTrend)
aSeasonality = aAddSpecialTokens(aSeasonality)
aKnown = aAddSpecialTokens(aKnown)
aObserved = aAddSpecialTokens(aObserved)


os.makedirs(TOKENIZIED_DATA_FOLDER)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\distribution.npy', aDistribution)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\dynamic_digits.npy', aDynamicDigits)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\trend.npy', aTrend)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\seasonality.npy', aSeasonality)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\known.npy', aKnown)
np.save(f'{TOKENIZIED_DATA_FOLDER}\\observed.npy', aObserved)