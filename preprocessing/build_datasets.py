import sys
sys.path.append( '/home/yunusemre/Time-Series-Forecasting/')
from preprocessing.constants import *

import pandas as pd

import numpy as np

'''
    inputs - expected to have data formatten in a dataframe with the columns of [target], [group_id] and [target_idx]

    outputs - 
        lb_train - (None, nr_of_lb_time_steps, nr_of_channels)
        fc_train - (None, nr_of_fc_time_steps, nr_of_channels)
        lb_test - (None, nr_of_lb_time_steps, nr_of_channels)
        lb_test - (None, nr_of_lb_time_steps, nr_of_channels)
'''
if __name__ == '__main__':


    dfTsDataset = pd.read_csv(
        f'{CONVERTED_DATA_FOLDER}/TimeSeriesDataset.csv',
        delimiter=';'
    )


    iNrOfLookbackSteps = LOOKBACK_COEFFICIENT*FORECAST_HORIZON
    iNrOfForecastSteps = FORECAST_HORIZON


    aTickerGroups = ['EURUSD_ticker', 'GBPUSD_ticker', 'USDCAD_ticker']
    aKnownGroups = []
    aObservedGroups = []



    aTimeIxs = np.arange(
        start= dfTsDataset.loc[:, 'time_idx'].to_numpy(int).min() + iNrOfLookbackSteps , 
        stop = dfTsDataset.loc[:, 'time_idx'].to_numpy(int).max(),
        step = iNrOfLookbackSteps + iNrOfForecastSteps
        )

    aTimeIxs = np.intersect1d(aTimeIxs , dfTsDataset.loc[:, 'time_idx'].to_numpy(int))


    # lookback tickers
    arr = []
    for s in aTickerGroups:

        df = dfTsDataset.query('group_id == @s').copy()

        arr2 = []
        for i in range(-iNrOfLookbackSteps,0):
            arr2.append(df.query('time_idx in (@aTimeIxs + @i)').loc[:, 'value'].to_numpy(dtype = np.float64))

        arr.append(arr2)

    arr = np.stack(arr )
    lb = np.transpose(arr , (2,1,0))



    # forecast tickers
    arr = []
    for s in aTickerGroups:

        df = dfTsDataset.query('group_id == @s').copy()

        arr2 = []
        for i in range(0,iNrOfForecastSteps):
            arr2.append(df.query('time_idx in (@aTimeIxs + @i)').loc[:, 'value'].to_numpy(dtype = np.float64))

        arr.append(arr2)

    arr = np.stack(arr )
    fc = np.transpose(arr , (2,1,0))

    



    print('done !')




