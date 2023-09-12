import numpy as np

import os

import pandas as pd

from settings import TRAINING_DATASETS_FOLDER, CONVERTED_DATA_FOLDER

import shutil

from utils import get_args_to_build_datasets


if __name__ == '__main__':
    '''
    builds numpy datasets based on the pandas dataframe of a given channel \
        located in CONVERTED_DATA_FOLDER folder.
    training and test datasets are split.

    inputs - expected to have data formatten in a dataframe with the \
        columns of [target], [group_id] and [target_idx]
        in the folder CONVERTED_DATA_FOLDER.

    outputs - produces 6 dataset for the given channel.
        lb_train: (None, lookback_horizon)
        fc_train: (None, forecast_horizon)
        ix_train: (None)

        lb_test: (None, lookback_horizon)
        fc_test: (None, forecast_horizon)
        ix_test: (None)
    '''

    args = get_args_to_build_datasets()

    target_group = args.target_group
    lookback_horizon = args.lookback_coefficient*args.forecast_horizon
    forecast_horizon = args.forecast_horizon
    test_size = args.test_size
    step_size = args.step_size
    channel = args.channel

    sSubDirectory = os.path.join(TRAINING_DATASETS_FOLDER, channel)
    if os.path.exists(sSubDirectory) is True:
        shutil.rmtree(sSubDirectory)

    dfTsDataset = pd.read_csv(
        os.path.join(CONVERTED_DATA_FOLDER, f'{channel}.csv'),
        delimiter=';')

    aTimeIxs = np.arange(
        start=dfTsDataset.loc[:, 'time_idx'].to_numpy(int).min() +
        lookback_horizon,
        stop=dfTsDataset.loc[:, 'time_idx'].to_numpy(int).max(),
        step=step_size
        )

    aTimeIxs = np.intersect1d(
        aTimeIxs,
        dfTsDataset.loc[:, 'time_idx'].to_numpy(int))
    dfSearch = dfTsDataset.query(
        '(group_id == @target_group)')[['time_idx', 'value']].copy()

    dfLb = pd.DataFrame(index=aTimeIxs)
    for i in range(-lookback_horizon, 0):
        dfFound = dfSearch.query('time_idx in (@aTimeIxs + @i)').copy()
        aFound = dfFound.loc[:, 'time_idx'].to_numpy() - i

        dfLb.loc[aFound, i] = dfFound.loc[:, 'value'].to_numpy()

    dfLb.dropna(inplace=True)

    dfFc = pd.DataFrame(index=aTimeIxs)
    for i in range(0, forecast_horizon):
        dfFound = dfSearch.query('time_idx in (@aTimeIxs + @i)').copy()
        aFound = dfFound.loc[:, 'time_idx'].to_numpy() - i

        dfFc.loc[aFound, i] = dfFound.loc[:, 'value'].to_numpy()

    dfFc.dropna(inplace=True)

    ixCommon = np.intersect1d(dfLb.index, dfFc.index)
    dfLb = dfLb.loc[ixCommon]
    dfFc = dfFc.loc[ixCommon]

    lb = dfLb.to_numpy()
    fc = dfFc.to_numpy()
    ix = dfLb.index.to_numpy()

    os.makedirs(sSubDirectory)
    np.save(os.path.join(sSubDirectory, 'lb_train.npy'), lb[:-test_size])
    np.save(os.path.join(sSubDirectory, 'fc_train.npy'), fc[:-test_size])
    np.save(os.path.join(sSubDirectory, 'ix_train.npy'), ix[:-test_size])

    np.save(os.path.join(sSubDirectory, 'lb_test.npy'), lb[-test_size:])
    np.save(os.path.join(sSubDirectory, 'fc_test.npy'), fc[-test_size:])
    np.save(os.path.join(sSubDirectory, 'ix_test.npy'), ix[-test_size:])

    print(f'Datasets for {channel} is built successfully.')
