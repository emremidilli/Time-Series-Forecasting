import numpy as np

import os

import pandas as pd

from settings import TRAINING_DATASETS_FOLDER, RAW_FREQUENCY_NUMPY, \
    DATETIME_FEATURES


def convert_unix_data(ts):
    ts = np.array(ts, dtype=f'datetime64[{RAW_FREQUENCY_NUMPY}]')

    srs = pd.Series(ts)  # noqa: F841

    converted = []
    for sFeature in DATETIME_FEATURES:
        arr = eval(f'srs.dt.{sFeature}')
        arr = arr.to_numpy()

        converted.append(arr)

    converted = np.stack(converted, axis=1)

    return converted


if __name__ == '__main__':
    sChannel = 'GBPUSD'

    training_data_dir = os.path.join(TRAINING_DATASETS_FOLDER, sChannel)

    ts_train = np.load(os.path.join(training_data_dir, 'ix_train.npy'))
    ts_test = np.load(os.path.join(training_data_dir, 'ix_test.npy'))

    ts_train = convert_unix_data(ts_train)
    ts_test = convert_unix_data(ts_test)

    np.save(os.path.join(training_data_dir, 'ts_train.npy'), ts_train)
    np.save(os.path.join(training_data_dir, 'ts_test.npy'), ts_test)
