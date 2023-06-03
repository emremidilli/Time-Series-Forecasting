'''
    Disadvantage(s):

        * It is not scalable at all if a new channel needs to be added. Whole training should be done from the begining.

    Inputs:

        * Scaled values - static covariates, knowns, observeds, distribution, dynamic digits, trend and seasonality for each channel.
        * Unscaled values - timestamps, distribution and deltas.

    Outputs:

        * Consolidated output under common timestamps.
'''

import numpy as np

import constants as c

import os

import shutil

CONVERTED_DATA_FOLDER = c.CONVERTED_DATA_FOLDER
CONSOLIDATED_CHANNEL_DATA_FOLDER = c.CONSOLIDATED_CHANNEL_DATA_FOLDER
SCALED_DATA_FOLDER= c.SCALED_DATA_FOLDER


if os.path.exists(CONSOLIDATED_CHANNEL_DATA_FOLDER) == True:
    shutil.rmtree(CONSOLIDATED_CHANNEL_DATA_FOLDER)

aFolderNames = os.listdir(CONVERTED_DATA_FOLDER)

dicTimeStamps = dict()
dicStaticCovariates = dict()
dicKnowns = dict()
dicObserveds = dict()
dicDistribution = dict()
dicDynamicDigits = dict()
dicTrend = dict()
dicSeasonality = dict()
dicQuantiles = dict()
dicDeltas= dict()

for sChannelId in aFolderNames:# aech sub-folder represents one channel.
    sUnscaledFolderName = f'{CONVERTED_DATA_FOLDER}\\{sChannelId}'
    sScaledFolderName = f'{SCALED_DATA_FOLDER}\\{sChannelId}'
    
    # from unscaled datasets
    dicTimeStamps[sChannelId] = np.load(f'{sUnscaledFolderName}\\timestamps.npy')
    dicDistribution[sChannelId] = np.load(f'{sUnscaledFolderName}\\distribution.npy')
    dicDeltas[sChannelId] = np.load(f'{sUnscaledFolderName}\\deltas.npy')
    
    
    # from unscaled datasets
    dicStaticCovariates[sChannelId] = np.load(f'{sScaledFolderName}\\static_covariates.npy')
    dicKnowns[sChannelId] = np.load(f'{sScaledFolderName}\\knowns.npy')
    dicObserveds[sChannelId] = np.load(f'{sScaledFolderName}\\observeds.npy')
    dicDynamicDigits[sChannelId] = np.load(f'{sScaledFolderName}\\dynamic_digits.npy')
    dicTrend[sChannelId] = np.load(f'{sScaledFolderName}\\trend.npy')
    dicSeasonality[sChannelId] = np.load(f'{sScaledFolderName}\\seasonality.npy')
    dicQuantiles[sChannelId] = np.load(f'{sScaledFolderName}\\quantiles.npy')
    
    
    
# find common time stamps on each channel Id
ix = None
for sChannelId in dicTimeStamps:
    if ix is None:
        ix = dicTimeStamps[sChannelId]
    else:
        ix = np.intersect1d(ix, dicTimeStamps[sChannelId])

        

for sChannelId in dicTimeStamps:
    
    ixSearch = dicTimeStamps[sChannelId]
    msk  = np.isin(ixSearch, ix)
    
    dicTimeStamps[sChannelId] = dicTimeStamps[sChannelId][msk]
    dicStaticCovariates[sChannelId] = np.expand_dims(dicStaticCovariates[sChannelId][msk], -1)
    dicKnowns[sChannelId] = np.expand_dims(dicKnowns[sChannelId][msk], -1)
    dicObserveds[sChannelId] = np.expand_dims(dicObserveds[sChannelId][msk], -1)
    dicDistribution[sChannelId] = np.expand_dims(dicDistribution[sChannelId][msk], -1)
    dicDynamicDigits[sChannelId] = np.expand_dims(dicDynamicDigits[sChannelId][msk], -1)
    dicTrend[sChannelId] = np.expand_dims(dicTrend[sChannelId][msk], -1)
    dicSeasonality[sChannelId] = np.expand_dims(dicSeasonality[sChannelId][msk], -1)
    dicQuantiles[sChannelId] = np.expand_dims(dicQuantiles[sChannelId][msk], -1)
    dicDeltas[sChannelId] = np.expand_dims(dicDeltas[sChannelId][msk], -1)
    
    
def aConvertDicToArray(dic):
    sFirstChannelId = list(dic.keys())[0]
    aToReturn =  dic[sFirstChannelId]

    i = 0
    for sChanneld in dic:
        if i != 0:
            aToReturn = np.append(aToReturn, dic[sChanneld], axis=-1)

        i = i + 1 
        
    return aToReturn



aTimestamps = dicTimeStamps[list(dicTimeStamps.keys())[0]]
aStaticCovariates = aConvertDicToArray(dicStaticCovariates)
aKnowns = aConvertDicToArray(dicKnowns)
aObserveds = aConvertDicToArray(dicObserveds)
aDistribution = aConvertDicToArray(dicDistribution)
aDynamicDigits = aConvertDicToArray(dicDynamicDigits)
aTrend = aConvertDicToArray(dicTrend)
aSeasonality = aConvertDicToArray(dicSeasonality)
aQuantiles =  aConvertDicToArray(dicQuantiles)
aDeltas =  aConvertDicToArray(dicDeltas)



os.makedirs(CONSOLIDATED_CHANNEL_DATA_FOLDER)
np.save(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\timestamps.npy', aTimestamps)
np.save(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\static_covariates.npy', aStaticCovariates)
np.save(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\knowns.npy', aKnowns)
np.save(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\observeds.npy', aObserveds)
np.save(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\distribution.npy', aDistribution)
np.save(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\dynamic_digits.npy', aDynamicDigits)
np.save(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\trend.npy', aTrend)
np.save(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\seasonality.npy', aSeasonality)
np.save(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\quantiles.npy', aQuantiles)
np.save(f'{CONSOLIDATED_CHANNEL_DATA_FOLDER}\\deltas.npy', aDeltas)