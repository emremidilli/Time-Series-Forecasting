'''
Used only to get the univarsal datasets.
'''

from datautils import get_dls

import numpy as np

import os

import shutil

from utils import get_args

if __name__ == '__main__':
    args = get_args()
    print(args)

    class Params:
        dset = args.input_dataset_id
        context_points = args.lookback_coefficient * args.forecast_horizon
        target_points = args.forecast_horizon
        batch_size = 128
        num_workers = 0
        with_ray = False
        features = args.features
        use_time_features = True
    params = Params
    dls = get_dls(params)

    lb_train = []
    fc_train = []
    ts_train = []

    for i, batch in enumerate(dls.train):
        lb_train.append(batch[0])
        fc_train.append(batch[1])
        ts_train.append(batch[2])

    for i, batch in enumerate(dls.valid):
        lb_train.append(batch[0])
        fc_train.append(batch[1])
        ts_train.append(batch[2])

    lb_train = np.concatenate(lb_train, axis=0)
    fc_train = np.concatenate(fc_train, axis=0)
    ts_train = np.concatenate(ts_train, axis=0)

    lb_test = []
    fc_test = []
    ts_test = []
    for i, batch in enumerate(dls.test):
        lb_test.append(batch[0])
        fc_test.append(batch[1])
        ts_test.append(batch[2])

    lb_test = np.concatenate(lb_test, axis=0)
    fc_test = np.concatenate(fc_test, axis=0)
    ts_test = np.concatenate(ts_test, axis=0)

    # use only time features of the last timestep of lookback window
    ts_train = ts_train[:, -1, :]
    ts_test = ts_test[:, -1, :]

    # for MS, forecasting horizons should be OT field.
    if args.features == 'MS':
        fc_train = fc_train[:, :, [-1]]
        fc_test = fc_test[:, :, [-1]]

    save_dir = os.path.join(
        os.environ.get('BIN_NAME'),
        os.environ.get('FORMATTED_NAME'),
        args.output_dataset_id)
    if os.path.exists(save_dir) is True:
        shutil.rmtree(save_dir)

    os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'lb_train.npy'), lb_train)
    np.save(os.path.join(save_dir, 'fc_train.npy'), fc_train)
    np.save(os.path.join(save_dir, 'ts_train.npy'), ts_train)

    np.save(os.path.join(save_dir, 'lb_test.npy'), lb_test)
    np.save(os.path.join(save_dir, 'fc_test.npy'), fc_test)
    np.save(os.path.join(save_dir, 'ts_test.npy'), ts_test)

    print(f'Completed.\tTrain-Test size: {len(lb_train)} - {len(lb_test)}')
