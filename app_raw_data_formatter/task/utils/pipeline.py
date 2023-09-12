import numpy as np

import pandas as pd


def convert_unix_data(ts, raw_freq_npy, datetime_features):
    '''converts unix timestamps to datetime_features.'''
    ts = np.array(ts, dtype=f'datetime64[{raw_freq_npy}]')

    srs = pd.Series(ts)  # noqa: F841

    converted = []
    for sFeature in datetime_features:
        arr = eval(f'srs.dt.{sFeature}')
        arr = arr.to_numpy()

        converted.append(arr)

    converted = np.stack(converted, axis=1)

    return converted
