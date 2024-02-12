
from datamodule import DataLoaders

import os

from pred_dataset import Dataset_ETT_minute, \
    Dataset_ETT_hour, Dataset_Custom

DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange']


def get_dls(params):

    assert params.dset in DSETS, \
        f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"

    if not hasattr(params, 'use_time_features'):
        params.use_time_features = False

    root_path = os.path.join(
        os.environ.get('BIN_NAME'),
        os.environ.get('RAW_DATA_NAME')
    )

    if params.dset == 'ettm1':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_ETT_minute,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
            batch_size=params.batch_size,
            workers=params.num_workers,
            )

    elif params.dset == 'ettm2':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_ETT_minute,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
            batch_size=params.batch_size,
            workers=params.num_workers,
            )

    elif params.dset == 'etth1':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_ETT_hour,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
            batch_size=params.batch_size,
            workers=params.num_workers,
            )

    elif params.dset == 'etth2':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_ETT_hour,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
            batch_size=params.batch_size,
            workers=params.num_workers,
            )

    elif params.dset == 'electricity':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
            batch_size=params.batch_size,
            workers=params.num_workers,
            )

    elif params.dset == 'traffic':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
            batch_size=params.batch_size,
            workers=params.num_workers,
            )

    elif params.dset == 'weather':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
            batch_size=params.batch_size,
            workers=params.num_workers,
            )

    elif params.dset == 'illness':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'national_illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
            batch_size=params.batch_size,
            workers=params.num_workers,
            )

    elif params.dset == 'exchange':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange_rate.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
            batch_size=params.batch_size,
            workers=params.num_workers,
            )
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls
