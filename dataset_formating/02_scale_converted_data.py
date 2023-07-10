'''
    Input:

        * Converted values except distribution. Because distribution is already between 0 and 1.

    Analysis:

        * MinMaxScaler is applied to scale the data between 0 and 1.

    Output:

        * Scaled datasets.
        * Scalers.
'''


from sklearn.preprocessing import MinMaxScaler
import joblib

import constants as c
import os

import shutil

import numpy as np


CONVERTED_DATA_FOLDER = c.CONVERTED_DATA_FOLDER 
SCALERS_FOLDER = c.SCALERS_FOLDER
SCALED_DATA_FOLDER = c.SCALED_DATA_FOLDER


# clean scaled data in case they exist
if os.path.exists(SCALED_DATA_FOLDER) == True:
    shutil.rmtree(SCALED_DATA_FOLDER)

os.makedirs(SCALED_DATA_FOLDER)


# create scalers folders in case it doesnt exist.
if os.path.exists(SCALERS_FOLDER) == False:
    os.makedirs(SCALERS_FOLDER)




def aScale(arr, sScalerId):
    oScaler= MinMaxScaler(clip = True)
    arr = oScaler.fit_transform(arr.reshape(-1, 1)).reshape(arr.shape)
    
    sScalerPath = f'{SCALERS_FOLDER}\\{sScalerId}.save'
    
    if os.path.exists(sScalerPath) == True:
        os.remove(sScalerPath)
        
    joblib.dump(oScaler, sScalerPath) # save scaler

    return arr


aChannelIds = os.listdir(CONVERTED_DATA_FOLDER)
for sChannelId in aChannelIds:
    sFolderName = f'{CONVERTED_DATA_FOLDER}\\{sChannelId}'
    
    aStaticCovariates = np.load(f'{sFolderName}\\static_covariates.npy')

    # static covariate consists of 2 parts: digit and transition. Both of them are scaled with seperate scalers.
    aStaticCovariates[: ,[0]] =  aScale(aStaticCovariates[: ,[0]], f'{sChannelId}_static_digits')
    aStaticCovariates[: ,[1]] =  aScale(aStaticCovariates[: ,[1]], f'{sChannelId}_transitions')
    
    # observeds
    aObserveds = np.load(f'{sFolderName}\\observeds.npy')
    aObserveds = aScale(aObserveds, f'{sChannelId}_observeds')
    
    # knowns
    aKnowns = np.load(f'{sFolderName}\\knowns.npy')
    aKnowns = aScale(aKnowns, f'{sChannelId}_knowns')    
    
    # dynamic digits
    aDynamicDigits = np.load(f'{sFolderName}\\dynamic_digits.npy')
    aDynamicDigits = aScale(aDynamicDigits, f'{sChannelId}_dynamic_digits')
    
    # trend
    aTrend = np.load(f'{sFolderName}\\trend.npy')
    aTrend = aScale(aTrend, f'{sChannelId}_trend')
    
    # seasonality
    aSeasonality = np.load(f'{sFolderName}\\seasonality.npy')
    aSeasonality = aScale(aSeasonality, f'{sChannelId}_seasonality')
    
    # quantiles
    aQuantiles = np.load(f'{sFolderName}\\quantiles.npy')
    aQuantiles = aScale(aQuantiles, f'{sChannelId}_quantiles')
    
    # deltas
    aDeltas = np.load(f'{sFolderName}\\deltas.npy')
    aDeltas = aScale(aDeltas, f'{sChannelId}_deltas')
    

    sSubFolder = f'{SCALED_DATA_FOLDER}\\{sChannelId}'
    os.makedirs(sSubFolder)
    
    np.save(f'{sSubFolder}\\static_covariates.npy', aStaticCovariates)
    np.save(f'{sSubFolder}\\knowns.npy', aKnowns)
    np.save(f'{sSubFolder}\\observeds.npy', aObserveds)
    np.save(f'{sSubFolder}\\dynamic_digits.npy', aDynamicDigits)
    np.save(f'{sSubFolder}\\trend.npy', aTrend)
    np.save(f'{sSubFolder}\\seasonality.npy', aSeasonality)
    np.save(f'{sSubFolder}\\quantiles.npy', aQuantiles)
    np.save(f'{sSubFolder}\\deltas.npy', aDeltas)