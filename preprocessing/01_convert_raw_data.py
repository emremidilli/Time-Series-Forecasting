import pandas as pd

import numpy as np

import constants as c

import os

import shutil

from sklearn.preprocessing import KBinsDiscretizer

from tensorflow.keras.layers import AveragePooling1D 


RAW_DATA_FOLDER = c.RAW_DATA_FOLDER
RAW_FREQUENCY = c.RAW_FREQUENCY
PATCH_SIZE= c.PATCH_SIZE
FORECAST_HORIZON = c.FORECAST_HORIZON
LOOKBACK_COEFFICIENT = c.LOOKBACK_COEFFICIENT
THRESHOLD_STATIC_SENSITIVITY = c.THRESHOLD_STATIC_SENSITIVITY

NR_OF_BINS = c.NR_OF_BINS

PATCH_SAMPLE_RATE = c.PATCH_SAMPLE_RATE
POOL_SIZE = c.POOL_SIZE

DATETIME_FEATURES = c.DATETIME_FEATURES
CONVERTED_DATA_FOLDER = c.CONVERTED_DATA_FOLDER 
TARGET_QUANTILES = c.TARGET_QUANTILES


iNrOfLookbackPatches = int((FORECAST_HORIZON*LOOKBACK_COEFFICIENT)/PATCH_SIZE)
iNrOfForecastPatches = int(FORECAST_HORIZON/PATCH_SIZE)

aLookbackTimeSteps = list(range(-(LOOKBACK_COEFFICIENT*FORECAST_HORIZON) , 0))
aForecastTimeSteps = list(range(1, FORECAST_HORIZON + 1))



if os.path.exists(CONVERTED_DATA_FOLDER) == True:
    shutil.rmtree(CONVERTED_DATA_FOLDER)

os.makedirs(CONVERTED_DATA_FOLDER)

aFileNames = os.listdir(RAW_DATA_FOLDER)
for sFileName in aFileNames:
    
    print(f'Converting file {sFileName}')

    dfRaw = pd.read_csv(
        f'{RAW_DATA_FOLDER}\\{sFileName}', 
        delimiter='\t',
        usecols=['<DATE>', '<TIME>','<HIGH>', '<VOL>']
    )


    dfRaw.loc[:, 'TIME_STAMP'] = dfRaw.loc[:, '<DATE>'] + ' ' +dfRaw.loc[:, '<TIME>']
    dfRaw.drop(['<DATE>', '<TIME>'], axis = 1, inplace = True)


    dfRaw.rename(columns = {'<HIGH>':'TICKER',
                         '<VOL>':'OBSERVED'
                        }, inplace = True)
    
    
    ixOutlierVolume = dfRaw.query('OBSERVED > 180000000').index
    dfRaw.loc[ixOutlierVolume, 'OBSERVED'] = 180000000
    dfRaw.loc[:, 'TIME_STAMP'] = pd.to_datetime(dfRaw.loc[:,'TIME_STAMP'])
    
    
    
    # calculate static and dynamic digits of tickers
    def truncate(n, decimals=0):
        decimals = int(decimals)
        multiplier = 10 ** decimals
        return ((n * multiplier).astype(int)) / multiplier

    # find static digit with finest granual format
    iMaxNrOfDecimals = 5 #  ??? should calculate automatically
    iDecimals = iMaxNrOfDecimals

    while True:
        converted = truncate(dfRaw.loc[:,'TICKER'],iDecimals)

        aDiff=  converted.diff().dropna()

        aDiff[aDiff!=0] = 1
        aDiff= aDiff.astype(int)

        fSensitivity = (aDiff.sum()/aDiff.shape[0])


        if fSensitivity <=THRESHOLD_STATIC_SENSITIVITY:
            break
        else:
            iDecimals = iDecimals - 1


    dfRaw.loc[:,'STATIC_TICKER'] = truncate(dfRaw.loc[:,'TICKER'],iDecimals)
    dfRaw.loc[:, 'DYNAMIC_TICKER'] = (dfRaw.loc[:, 'TICKER'] - dfRaw.loc[:, 'STATIC_TICKER'])
    
    

    ix = pd.date_range(
        start = dfRaw.loc[:, 'TIME_STAMP'].min()+ pd.Timedelta(f'{-1}{RAW_FREQUENCY}'),
        end = dfRaw.loc[:, 'TIME_STAMP'].max()+ pd.Timedelta(f'{-1}{RAW_FREQUENCY}'),
        freq=f'{PATCH_SIZE}{RAW_FREQUENCY}'
    )



    dfObserveds = pd.DataFrame(index = ix, columns  = (aLookbackTimeSteps + aForecastTimeSteps), dtype = 'float64')
    dfLookbackTickers = pd.DataFrame(index = ix, columns  = aLookbackTimeSteps, dtype = 'float64')
    for i in aLookbackTimeSteps:
        ixSearch = ix + pd.Timedelta(f'{i}{RAW_FREQUENCY}')

        ### ???  search by considering weekend & vacations
        dfFound = dfRaw.query('TIME_STAMP in @ixSearch')

        msk = np.in1d(ixSearch.to_numpy(),dfFound.loc[:, 'TIME_STAMP'].to_numpy())

        dfLookbackTickers.loc[msk, i] = dfFound.loc[:, 'TICKER'].to_numpy(dtype = 'float64')
        dfObserveds.loc[msk, i] = dfFound.loc[:, 'OBSERVED'].to_numpy(dtype = 'float64')


        
    dfForecastTickers = pd.DataFrame(index = ix, columns  = aForecastTimeSteps, dtype = 'float64')
    for i in aForecastTimeSteps:
        ixSearch = ix + pd.Timedelta(f'{i}{RAW_FREQUENCY}')

        ### ???  search by considering weekend & vacations
        dfFound = dfRaw.query('TIME_STAMP in @ixSearch')

        msk = np.in1d(ixSearch.to_numpy(),dfFound.loc[:, 'TIME_STAMP'].to_numpy())

        dfForecastTickers.loc[msk, i] = dfFound.loc[:, 'TICKER'].to_numpy(dtype = 'float64')
        dfObserveds.loc[msk, i] = dfFound.loc[:, 'OBSERVED'].to_numpy(dtype = 'float64')

        
        


    # handle missing data
    dfLookbackTickers.dropna(inplace = True)
    dfForecastTickers.dropna(inplace = True)
    dfObserveds.dropna(inplace = True)
    ix = np.intersect1d(dfLookbackTickers.index, dfForecastTickers.index)
    ix = np.intersect1d(ix, dfObserveds.index)

    dfObserveds = dfObserveds.loc[ix]
    dfLookbackTickers = dfLookbackTickers.loc[ix]
    dfForecastTickers = dfForecastTickers.loc[ix]
    dfWholeTickers = dfLookbackTickers.merge(right= dfForecastTickers, left_index = True, right_index = True, how=  'inner')

    # identify static digits
    dfStaticDigits = truncate(dfLookbackTickers, iDecimals)

    # identify number of transitions
    dfTransitions = ((dfStaticDigits.max(axis = 1) - dfStaticDigits.min(axis = 1))/(10 ** -iDecimals)).astype(int).to_frame()
    dfTransitions.rename(columns = {0:'NR_OF_TRANSITIONS'}, inplace = True)
    
    # identify dynamic digits
    dfLookbackDynamicDigits = dfLookbackTickers.sub(dfStaticDigits.min(axis = 1),  axis = 0)
    dfForecastDynamicDigits = dfForecastTickers.sub(dfStaticDigits.min(axis = 1),  axis = 0)
    dfWholeDynamicDigits = dfLookbackDynamicDigits.merge(right = dfForecastDynamicDigits, left_index = True, right_index = True, how = 'inner')


    # static covariates
    dfStaticCovariates = dfStaticDigits.min(axis = 1).to_frame()
    dfStaticCovariates.rename(columns = {0:'STATIC_DIGIT'}, inplace = True)
    dfStaticCovariates = dfStaticCovariates.merge(right = dfTransitions, left_index = True, right_index = True, how = 'inner')
    arr = dfStaticCovariates.to_numpy(dtype = 'float64')
    aStaticCovariates = arr.copy()
    
    
    # observeds
    arr = dfObserveds.to_numpy(dtype = 'float64')
    arr = np.reshape(arr , (arr.shape[0], iNrOfLookbackPatches + iNrOfForecastPatches ,-1))
    arr = np.quantile(arr , [0.1, 0.9], axis = 2)
    arr =np.transpose(arr ,(1, 2, 0))
    aObserveds = arr.copy()
    
    
    # knowns
    ix = dfWholeTickers.index
    arr=  np.zeros(shape = (ix.shape[0], iNrOfLookbackPatches + iNrOfForecastPatches, len(DATETIME_FEATURES) ))

    for i in range(-iNrOfLookbackPatches, iNrOfForecastPatches):
        ixSearch = ix + pd.Timedelta(f'{i*PATCH_SIZE}{RAW_FREQUENCY}')

        j = 0
        for sDatePart in DATETIME_FEATURES:
            exec(f'arr[:, {i} ,{j}] = ixSearch.{sDatePart}')

            j = j + 1
    aKnowns = arr.copy()
    
    
    
    # distribution representations
    dfLookbackNormalization = (dfWholeTickers.sub(dfLookbackTickers.min(axis = 1), axis = 0))
    dfLookbackNormalization = dfLookbackNormalization.div(dfLookbackTickers.max(axis = 1)- dfLookbackTickers.min(axis = 1), axis = 0) 
    
    dfLookbackNormalization[dfLookbackNormalization>1] = 1 #in case forecast horizon has increasing trend
    dfLookbackNormalization[dfLookbackNormalization<0] = 0 #in case forecast horizon has decreasing trend
    
    oBinsDiscretizer = KBinsDiscretizer(n_bins=NR_OF_BINS, encode = 'ordinal', strategy='uniform')
    arr = dfLookbackNormalization.to_numpy(dtype = 'float64')
    arr = np.transpose(arr)
    arr = oBinsDiscretizer.fit_transform(arr)
    arr = np.transpose(arr).astype(int)

    arr = np.reshape(arr , (arr.shape[0], iNrOfLookbackPatches + iNrOfForecastPatches ,-1))

    arr = np.apply_along_axis(lambda x: np.bincount(x, minlength=NR_OF_BINS), 2, arr) 

    arr =(arr/PATCH_SIZE) # probabilities
    
    arr = np.around(arr, decimals = 2) #rounded to 2
    
    aDistribution = arr.copy()
    
    
    # dynamic digit representation
    arr = dfWholeDynamicDigits.to_numpy(dtype = 'float64')
        # also should be normalized.
    arr = np.reshape(arr , (arr.shape[0], iNrOfLookbackPatches + iNrOfForecastPatches ,-1))

    arr = np.quantile(arr , [0.1, 0.5, 0.9], axis = 2)
    arr =np.transpose(arr ,(1, 2, 0))
    
    aDynamicDigits = arr.copy()
    
    
    # trend & seasonality
    arr = dfWholeTickers.to_numpy()
    arr = np.reshape(arr , (arr.shape[0], iNrOfLookbackPatches + iNrOfForecastPatches ,-1))
    arr2 = np.subtract(arr ,  np.expand_dims(np.min(arr , axis = 2), 2))
    arr2 = np.divide(arr2 ,  np.expand_dims(np.max(arr , axis = 2) - np.min(arr , axis = 2), 2))
    # there can be warning from previous np.divide due to 0/0. Warning can be ignored.
    # but it will cause nan.
    # nans are replaced with 0.5 which is the middle value of min-max normalization.
    arr2 [np.isnan(arr2 )] = 0.5 
    arr2 = np.reshape(arr2, (arr2.shape[0], - 1))
    dfSelfPatchNormalization = pd.DataFrame(data = arr2, index = dfWholeTickers.index, columns = dfWholeTickers.columns)

    arr = dfSelfPatchNormalization.to_numpy()
    arr = np.reshape(arr , (arr.shape[0], iNrOfLookbackPatches + iNrOfForecastPatches,-1))

    avg_pool_sampling = AveragePooling1D (pool_size=int(PATCH_SAMPLE_RATE * PATCH_SIZE),padding='same')
    arr  = np.transpose(arr,  (0,2,1))
    arr = np.around(arr, decimals = 2) #rounded to 2
    aSample = avg_pool_sampling(arr)

    avg_pool_trend = AveragePooling1D (pool_size=POOL_SIZE, strides = 1 ,padding='same')
    arr = avg_pool_trend(aSample)
    arr = np.around(arr, decimals = 2) #rounded to 2
    aTrend = arr.copy()
    
    aSample  = np.transpose(aSample,  (0,2,1))
    aTrend  = np.transpose(aTrend,  (0,2,1))
    
    arr = aSample - aTrend
    arr = np.around(arr, decimals = 2) #rounded to 2
    aSeasonality = arr.copy()
    

 
   # quantiles
    arr = dfForecastDynamicDigits.to_numpy(dtype = 'float64')
    arr = np.reshape(arr , (arr.shape[0], iNrOfForecastPatches, -1))
    arr = np.quantile(arr , TARGET_QUANTILES, axis = 2)
    arr =np.transpose(arr ,(1, 2, 0))
    aQuantiles = arr.copy()
    
    
    # deltas
    arr = dfWholeTickers.to_numpy(dtype = 'float64')
    arr = np.reshape(arr , (arr.shape[0], iNrOfLookbackPatches + iNrOfForecastPatches, -1))
    arr = np.quantile(arr , TARGET_QUANTILES, axis = 2)
    arr =np.transpose(arr ,(1, 2, 0))
    arrDiff = np.roll(arr,  shift = -1, axis=1)-arr #last position of axis 1 should be disregarded.
    arr= arrDiff/arr
    arr[:, -1,:] = 0 #last position of axis 1 should be disregarded. We set 0 for them to have standard shape.
    aDeltas = arr.copy()
    
    
    # timestamps
    aTimestamps = dfForecastTickers.index.to_numpy()
    

    sChannelId = sFileName.replace('.csv', '')
    sConvertedDataSubFolder = f'{CONVERTED_DATA_FOLDER}\\{sChannelId}'
    os.makedirs(sConvertedDataSubFolder)
    

    np.save(f'{sConvertedDataSubFolder}\\timestamps.npy', aTimestamps)
    np.save(f'{sConvertedDataSubFolder}\\static_covariates.npy', aStaticCovariates)
    np.save(f'{sConvertedDataSubFolder}\\knowns.npy', aKnowns)
    np.save(f'{sConvertedDataSubFolder}\\observeds.npy', aObserveds)
    np.save(f'{sConvertedDataSubFolder}\\distribution.npy', aDistribution)
    np.save(f'{sConvertedDataSubFolder}\\dynamic_digits.npy', aDynamicDigits)
    np.save(f'{sConvertedDataSubFolder}\\trend.npy', aTrend)
    np.save(f'{sConvertedDataSubFolder}\\seasonality.npy', aSeasonality)
    np.save(f'{sConvertedDataSubFolder}\\quantiles.npy', aQuantiles)
    np.save(f'{sConvertedDataSubFolder}\\deltas.npy', aDeltas)