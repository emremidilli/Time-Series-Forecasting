import sys
sys.path.append( '/home/yunusemre/Time-Series-Forecasting/')
from preprocessing.constants import *

import pandas as pd

import numpy as np

import os
import shutil


if __name__ == '__main__':
    '''
        inputs - expected to have data formatten in a dataframe with the columns of [target], [group_id] and [target_idx] 
            in the folder CONVERTED_DATA_FOLDER.

        outputs - for each channel in CONVERTED_DATA_FOLDER
            lb_train- (None, nr_of_lb_time_steps)
            fc_train - (None, nr_of_fc_time_steps)
            ix_train - (None)
            
            lb_test- (None, nr_of_lb_time_steps)
            fc_test - (None, nr_of_fc_time_steps)
            ix_test - (None)
    '''
    sTickerGroup = 'ticker'


    iNrOfLookbackSteps = LOOKBACK_COEFFICIENT*FORECAST_HORIZON
    iNrOfForecastSteps = FORECAST_HORIZON

    if os.path.exists(TRAINING_DATA_FOLDER) == True:
        shutil.rmtree(TRAINING_DATA_FOLDER)

    os.makedirs(TRAINING_DATA_FOLDER)

    for sFileName in os.listdir(CONVERTED_DATA_FOLDER):

        dfTsDataset = pd.read_csv(
            f'{CONVERTED_DATA_FOLDER}/{sFileName}',
            delimiter=';'
        )

        aTimeIxs = np.arange(
            start= dfTsDataset.loc[:, 'time_idx'].to_numpy(int).min() + iNrOfLookbackSteps , 
            stop = dfTsDataset.loc[:, 'time_idx'].to_numpy(int).max(),
            step = PATCH_SIZE
            )

        aTimeIxs = np.intersect1d(aTimeIxs , dfTsDataset.loc[:, 'time_idx'].to_numpy(int))
        dfSearch = dfTsDataset.query('(group_id == @sTickerGroup)')[['time_idx', 'value']].copy()

        
        dfLb = pd.DataFrame(index = aTimeIxs)
        for i in range(-iNrOfLookbackSteps,0):
            dfFound = dfSearch.query('time_idx in (@aTimeIxs + @i)').copy()
            aFound = dfFound.loc[:, 'time_idx'].to_numpy() - i

            dfLb.loc[aFound, i] = dfFound.loc[:, 'value'].to_numpy()

        dfLb.dropna(inplace = True)

        dfFc = pd.DataFrame(index = aTimeIxs)
        for i in range(0,iNrOfForecastSteps):
            dfFound = dfSearch.query('time_idx in (@aTimeIxs + @i)').copy()
            aFound = dfFound.loc[:, 'time_idx'].to_numpy() - i

            dfFc.loc[aFound, i] = dfFound.loc[:, 'value'].to_numpy()

        dfFc.dropna(inplace = True)


        ixCommon  = np.intersect1d(dfLb.index, dfFc.index)
        dfLb = dfLb.loc[ixCommon]
        dfFc = dfFc.loc[ixCommon]


        lb = dfLb.to_numpy()
        fc = dfFc.to_numpy()
        ix = dfLb.index.to_numpy()

        
        np.save(f'{TRAINING_DATA_FOLDER}/lb_train.npy', lb[:-TEST_SIZE])
        np.save(f'{TRAINING_DATA_FOLDER}/fc_train.npy', fc[:-TEST_SIZE])
        np.save(f'{TRAINING_DATA_FOLDER}/ix_train.npy', ix[:-TEST_SIZE])

        np.save(f'{TRAINING_DATA_FOLDER}/lb_test.npy', lb[-TEST_SIZE:])
        np.save(f'{TRAINING_DATA_FOLDER}/fc_test.npy', fc[-TEST_SIZE:])
        np.save(f'{TRAINING_DATA_FOLDER}/ix_test.npy', ix[-TEST_SIZE:])