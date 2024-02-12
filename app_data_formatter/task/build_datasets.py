import numpy as np

import os

import pandas as pd

import shutil

from utils import get_args_to_build_datasets, save_config_file, \
    convert_unix_data


if __name__ == '__main__':
    '''
    builds numpy datasets based on the pandas dataframe of a given
    list of covariates of a dataset located in
    CONVERTED_DATA_FOLDER folder.
    training and test datasets are split based on test_size.

    it is expected to have dataset in CONVERTED_DATA_FOLDER
    to be formatted as a dataframe with the
    columns of [time_idx], [group_id] and [value].
    this format is the TimeSeriesDataset format of pytorch forecasting.

    the script produces 6 numpy datasets in following formats:
    1. lb_train: (None, nr_of_covariates, lookback_horizon)
    2. fc_train: (None, nr_of_covariates, forecast_horizon)
    3. lb_test: (None, nr_of_covariates, lookback_horizon)
    4. fc_test: (None, nr_of_covariates, forecast_horizon)
    5. ts_train: (None, nr_of_timestamp_features)
    6. ts_test: (None, nr_of_timestamp_features)
    '''

    args = get_args_to_build_datasets()

    model_id = args.model_id
    dataset_id = args.dataset_id
    list_of_covariates = args.list_of_covariates
    forecast_horizon = args.forecast_horizon
    lookback_horizon = args.lookback_coefficient * forecast_horizon
    step_size = args.step_size
    test_size = args.test_size
    raw_frequency = args.raw_frequency
    datetime_features = args.datetime_features

    save_dir = os.path.join(
        os.environ.get('BIN_NAME'),
        os.environ.get('FORMATTED_NAME'),
        model_id)
    if os.path.exists(save_dir) is True:
        shutil.rmtree(save_dir)

    os.makedirs(save_dir)

    save_config_file(
        folder_dir=save_dir,
        args=args)

    dfTsDataset = pd.read_csv(
        os.path.join(
            os.environ.get('BIN_NAME'),
            os.environ.get('CONVERTED_DATA_NAME'),
            f'{dataset_id}.csv'),
        delimiter=';')

    aTimeIxs = np.arange(
        start=dfTsDataset.loc[:, 'time_idx'].to_numpy(int).min() +
        lookback_horizon,
        stop=dfTsDataset.loc[:, 'time_idx'].to_numpy(int).max(),
        step=step_size)

    aTimeIxs = np.intersect1d(
        aTimeIxs,
        dfTsDataset.loc[:, 'time_idx'].to_numpy(int))

    lookback_steps = range(-lookback_horizon, 0)
    forecast_steps = range(0, forecast_horizon)

    all_df_lb = dict()
    all_df_fc = dict()

    common_ix = None

    for covariate in list_of_covariates:

        dfSearch = dfTsDataset.\
            query('(group_id == @covariate)')[['time_idx', 'value']].copy()

        df_lb = pd.DataFrame(
            index=aTimeIxs,
            columns=lookback_steps)
        for i in lookback_steps:
            dfFound = dfSearch.query('time_idx in (@aTimeIxs + @i)').copy()
            aFound = dfFound.loc[:, 'time_idx'].to_numpy() - i

            df_lb.loc[aFound, i] = dfFound.loc[:, 'value'].to_numpy()

        df_lb.dropna(inplace=True)

        df_fc = pd.DataFrame(
            index=aTimeIxs,
            columns=forecast_steps)
        for i in forecast_steps:
            dfFound = dfSearch.query('time_idx in (@aTimeIxs + @i)').copy()
            aFound = dfFound.loc[:, 'time_idx'].to_numpy() - i

            df_fc.loc[aFound, i] = dfFound.loc[:, 'value'].to_numpy()

        df_fc.dropna(inplace=True)

        ix_cov = np.intersect1d(df_lb.index, df_fc.index)

        if common_ix is None:
            common_ix = ix_cov
        else:
            common_ix = np.intersect1d(common_ix, ix_cov)

        df_lb = df_lb.loc[common_ix]
        df_fc = df_fc.loc[common_ix]

        all_df_lb[covariate] = df_lb
        all_df_fc[covariate] = df_fc

        del df_lb, df_fc, ix_cov

    lb = None
    fc = None

    ix = common_ix

    for covariate in list_of_covariates:
        lb_to_add = np.expand_dims(
            all_df_lb[covariate].loc[common_ix].to_numpy(),
            axis=2)

        fc_to_add = np.expand_dims(
            all_df_fc[covariate].loc[common_ix].to_numpy(),
            axis=2)

        if lb is None:
            lb = lb_to_add
            fc = fc_to_add
        else:
            lb = np.concatenate((lb, lb_to_add), axis=2)
            fc = np.concatenate((fc, fc_to_add), axis=2)

    test_size = int(len(lb) * test_size)
    train_size = int(len(lb)) - test_size

    np.save(
        os.path.join(save_dir, 'ts_train.npy'),
        convert_unix_data(
            ix[:-test_size],
            raw_frequency,
            datetime_features))
    np.save(
        os.path.join(save_dir, 'ts_test.npy'),
        convert_unix_data(
            ix[-test_size:],
            raw_frequency,
            datetime_features))

    np.save(os.path.join(save_dir, 'lb_train.npy'), lb[:-test_size])
    np.save(os.path.join(save_dir, 'fc_train.npy'), fc[:-test_size])

    np.save(os.path.join(save_dir, 'lb_test.npy'), lb[-test_size:])
    np.save(os.path.join(save_dir, 'fc_test.npy'), fc[-test_size:])

    print(f'Completed.\tTrain-Test size: {train_size} - {test_size}')
