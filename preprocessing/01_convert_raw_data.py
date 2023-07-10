import pandas as pd

import numpy as np

from constants import *

import os

import shutil








'''
    input: original raw files

    output: 
        1. tickers (None, nr_of_lookback_steps + nr_of_forecast_tickers, 1)
        2. observeds (None, nr_of_lookback_steps + nr_of_forecast_tickers, 1)
        3. knowns (None, nr_of_lookback_steps + nr_of_forecast_tickers, nr_of_datetime_features)
        4. static covariates (None, 2) {Static ticker, Number of transition}
'''



aLookbackTimeSteps = list(range(-(LOOKBACK_COEFFICIENT*FORECAST_HORIZON) , 0))
aForecastTimeSteps = list(range(1, FORECAST_HORIZON + 1))
iNrOfLookbackPatches = int((FORECAST_HORIZON*LOOKBACK_COEFFICIENT)/PATCH_SIZE)
iNrOfForecastPatches = int(FORECAST_HORIZON/PATCH_SIZE)


if os.path.exists(CONVERTED_DATA_FOLDER) == True:
    shutil.rmtree(CONVERTED_DATA_FOLDER)

os.makedirs(CONVERTED_DATA_FOLDER)

aFileNames = os.listdir(RAW_DATA_FOLDER)




def aGetCommonTimeStampsAccrossChannels():

    aCommonTimeStamps = None
    for sFileName in aFileNames:
        dfRaw = pd.read_csv(
            f'{RAW_DATA_FOLDER}/{sFileName}', 
            delimiter='\t',
            usecols=['<DATE>', '<TIME>','<HIGH>', '<VOL>']
            )


        dfRaw.loc[:, 'TIME_STAMP'] = dfRaw.loc[:, '<DATE>'] + ' ' +dfRaw.loc[:, '<TIME>']
        dfRaw.loc[:, 'TIME_STAMP'] = pd.to_datetime(dfRaw.loc[:,'TIME_STAMP'])

        if aCommonTimeStamps is None:
            aCommonTimeStamps = dfRaw.loc[:, 'TIME_STAMP'].to_numpy(np.datetime64)
        else:
            aCommonTimeStamps = np.intersect1d(aCommonTimeStamps, dfRaw.loc[:, 'TIME_STAMP'].to_numpy(np.datetime64))

    return 


for sFileName in aFileNames:
    
    print(f'Converting file {sFileName}')

    sChannelId = sFileName.replace('.csv', '')
    sChannelName = sChannelId[:6]

    dfRaw = pd.read_csv(
        f'{RAW_DATA_FOLDER}/{sFileName}', 
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

    dfRaw.query('TIME_STAMP in @aCommonTimeStamps', inplace = True)


    
    
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
    




    

    ix = pd.date_range(
        start = dfRaw.loc[:, 'TIME_STAMP'].min()+ pd.Timedelta(f'{-1}{RAW_FREQUENCY}'),
        end = dfRaw.loc[:, 'TIME_STAMP'].max()+ pd.Timedelta(f'{-1}{RAW_FREQUENCY}'),
        freq=f'{FORECAST_HORIZON * (LOOKBACK_COEFFICIENT + 1)}{RAW_FREQUENCY}'
    )



    
    dfObserveds = pd.DataFrame(index = ix, columns  = (aLookbackTimeSteps + aForecastTimeSteps), dtype = 'float64')
    dfLookbackTickers = pd.DataFrame(index = ix, columns  = aLookbackTimeSteps, dtype = 'float64')

    dfTimeStamps = pd.DataFrame(index = ix, columns  = (aLookbackTimeSteps + aForecastTimeSteps))

    for i in aLookbackTimeSteps:
        ixSearch = ix + pd.Timedelta(f'{i}{RAW_FREQUENCY}')

        ### ???  search by considering weekend & vacations
        dfFound = dfRaw.query('TIME_STAMP in @ixSearch')

        msk = np.isin(ixSearch.to_numpy(dtype = ix.dtype),dfFound.loc[:, 'TIME_STAMP'].to_numpy(dtype = ix.dtype))

        dfLookbackTickers.loc[msk, i] = dfFound.loc[:, 'TICKER'].to_numpy(dtype = 'float64')
        dfObserveds.loc[msk, i] = dfFound.loc[:, 'OBSERVED'].to_numpy(dtype = 'float64')

        dfTimeStamps.loc[msk, i] = dfFound.loc[:, 'TIME_STAMP'].to_numpy(ix.dtype)


    dfForecastTickers = pd.DataFrame(index = ix, columns  = aForecastTimeSteps, dtype = 'float64')
    for i in aForecastTimeSteps:
        ixSearch = ix + pd.Timedelta(f'{i}{RAW_FREQUENCY}')

        ### ???  search by considering weekend & vacations
        dfFound = dfRaw.query('TIME_STAMP in @ixSearch')

        msk = np.isin(ixSearch.to_numpy(dtype = ix.dtype),dfFound.loc[:, 'TIME_STAMP'].to_numpy(dtype = ix.dtype))

        dfForecastTickers.loc[msk, i] = dfFound.loc[:, 'TICKER'].to_numpy(dtype = 'float64')
        dfObserveds.loc[msk, i] = dfFound.loc[:, 'OBSERVED'].to_numpy(dtype = 'float64')


        dfTimeStamps.loc[msk, i] = dfFound.loc[:, 'TIME_STAMP'].to_numpy(ix.dtype)

        
        
    


    # handle missing data
    dfLookbackTickers.dropna(inplace = True)
    dfForecastTickers.dropna(inplace = True)
    dfObserveds.dropna(inplace = True)
    dfTimeStamps.dropna(inplace = True)

    ix = np.intersect1d(dfLookbackTickers.index, dfForecastTickers.index)
    ix = np.intersect1d(ix, dfObserveds.index)
    ix = np.intersect1d(ix, dfTimeStamps.index)

    dfObserveds = dfObserveds.loc[ix]
    dfLookbackTickers = dfLookbackTickers.loc[ix]
    dfForecastTickers = dfForecastTickers.loc[ix]
    dfTimeStamps = dfTimeStamps.loc[ix]
    dfTickers = dfLookbackTickers.merge(right= dfForecastTickers, left_index = True, right_index = True, how=  'inner')



    dfTsDataset  = pd.DataFrame(columns = ['value', 'group_id', 'time_idx'])
    aTimeStamps = np.reshape(dfTimeStamps.to_numpy(dtype = np.datetime64), (-1,))
    aTickers = np.reshape(dfTickers.to_numpy(dtype = np.float64), (-1,))
    aObserveds = np.reshape(dfObserveds.to_numpy(dtype = np.float64), (-1,))
    
    df = pd.DataFrame(
        data = {
            'value': aTickers, 
            'group_id': f'{sChannelName}_ticker' ,
            'time_idx':aTimeStamps
            }
        )
    dfTsDataset = pd.concat([dfTsDataset, df])

    df = pd.DataFrame(
        data = {
            'value': aObserveds, 
            'group_id': f'{sChannelName}_observed' ,
            'time_idx':aTimeStamps
            }
        )


    
    for sDatePart in DATETIME_FEATURES:
        arr = None
        exec(f'arr = pd.DatetimeIndex(aTimeStamps).{sDatePart}')

        df = pd.DataFrame(
            data = {
                'value': arr, 
                'group_id': f'{sChannelName}_{sDatePart}' ,
                'time_idx':aTimeStamps
                }
            )
        
        dfTsDataset = pd.concat([dfTsDataset, df])