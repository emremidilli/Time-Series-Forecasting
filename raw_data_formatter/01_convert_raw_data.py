import sys
sys.path.append( '/home/yunusemre/Time-Series-Forecasting/')
from preprocessing.constants import *

import pandas as pd

import numpy as np

import os

import shutil



'''
    reads the channel files and identifies the common timestamps accross each channel.

    inputs - raw data in channel files.
    returns - common time stamps in format of numpy array.
'''
def aGetCommonTimeStampsAccrossChannels():

    arr = None
    aFileNames = os.listdir(RAW_DATA_FOLDER)
    for sFileName in aFileNames:
        dfRaw = pd.read_csv(
            f'{RAW_DATA_FOLDER}/{sFileName}', 
            delimiter='\t',
            usecols=['<DATE>', '<TIME>']
            )


        dfRaw.loc[:, 'TIME_STAMP'] = dfRaw.loc[:, '<DATE>'] + ' ' +dfRaw.loc[:, '<TIME>']
        dfRaw.loc[:, 'TIME_STAMP'] = pd.to_datetime(dfRaw.loc[:,'TIME_STAMP'])

        if arr is None:
            arr = dfRaw.loc[:, 'TIME_STAMP'].to_numpy(np.datetime64)
        else:
            arr = np.intersect1d(arr, dfRaw.loc[:, 'TIME_STAMP'].to_numpy(np.datetime64))

    return arr

'''
    identifies time stamps in way that they don't contain any gap within lookback and forecast horizons.
    in case a gap exists, that horizon is dropped.
    converts the dataset into a dataframe with the columns of [value], [group_id] and [time_idx]

    returns - converted pandas dataframe.
'''
def dfConvertToTimeSeriesDataset(aCommonTimeStamps):

    dfTsDataset  = pd.DataFrame(columns = ['value', 'group_id', 'time_idx'])

    aLookbackTimeSteps = list(range(-(LOOKBACK_COEFFICIENT*FORECAST_HORIZON) , 0))
    aForecastTimeSteps = list(range(0, FORECAST_HORIZON))
    aFileNames = os.listdir(RAW_DATA_FOLDER)
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
        dfTsDataset = pd.concat([dfTsDataset, df])


        
        for sDatePart in DATETIME_FEATURES:
            arr = pd.DatetimeIndex(aTimeStamps)
            lcls = locals()
            exec(f'arr = arr.{sDatePart}', globals(),lcls )
            arr = lcls['arr']
            
            df = pd.DataFrame(
                data = {
                    'value': arr, 
                    'group_id': f'{sChannelName}_{sDatePart}' ,
                    'time_idx':aTimeStamps
                    }
                )
            
            dfTsDataset = pd.concat([dfTsDataset, df])


    return dfTsDataset



if __name__ == '__main__' :
    if os.path.exists(CONVERTED_DATA_FOLDER) == True:
        shutil.rmtree(CONVERTED_DATA_FOLDER)

    aCommonTxs = aGetCommonTimeStampsAccrossChannels()
    dfTsDataset = dfConvertToTimeSeriesDataset(aCommonTxs)

    arr = dfTsDataset.loc[:, 'time_idx'].to_numpy(dtype = np.datetime64)
    arr = arr - arr.min()
    arr= (arr / np.timedelta64(1, RAW_FREQUENCY_NUMPY)).astype(int)
    dfTsDataset.loc[:, 'time_idx'] = arr


    os.makedirs(CONVERTED_DATA_FOLDER)

    dfTsDataset.to_csv(f'{CONVERTED_DATA_FOLDER}/TimeSeriesDataset.csv', index= None, sep = ';')