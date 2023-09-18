import numpy as np

import os

import pandas as pd

import shutil

from utils import get_args_to_build_datasets, save_config_file


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

    sub_directory = os.path.join(
        os.environ.get('BIN_NAME'),
        os.environ.get('FORMWATTED_NAME'),
        channel)
    if os.path.exists(sub_directory) is True:
        shutil.rmtree(sub_directory)

    os.makedirs(sub_directory)

    save_config_file(
        folder_dir=sub_directory,
        args=args)

    dfTsDataset = pd.read_csv(
        os.path.join(
            os.environ.get('BIN_NAME'),
            os.environ.get('CONVERTED_DATA_NAME'),
            f'{channel}.csv'),
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

    lookback_steps = range(-lookback_horizon, 0)
    dfLb = pd.DataFrame(
        index=aTimeIxs,
        columns=lookback_steps)
    for i in lookback_steps:
        dfFound = dfSearch.query('time_idx in (@aTimeIxs + @i)').copy()
        aFound = dfFound.loc[:, 'time_idx'].to_numpy() - i

        dfLb.loc[aFound, i] = dfFound.loc[:, 'value'].to_numpy()

    dfLb.dropna(inplace=True)

    forecast_steps = range(0, forecast_horizon)
    dfFc = pd.DataFrame(
        index=aTimeIxs,
        columns=forecast_steps)
    for i in forecast_steps:
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

    np.save(os.path.join(sub_directory, 'lb_train.npy'), lb[:-test_size])
    np.save(os.path.join(sub_directory, 'fc_train.npy'), fc[:-test_size])
    np.save(os.path.join(sub_directory, 'ix_train.npy'), ix[:-test_size])

    np.save(os.path.join(sub_directory, 'lb_test.npy'), lb[-test_size:])
    np.save(os.path.join(sub_directory, 'fc_test.npy'), fc[-test_size:])
    np.save(os.path.join(sub_directory, 'ix_test.npy'), ix[-test_size:])
