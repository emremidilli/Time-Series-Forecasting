import numpy as np

import os

from utils import get_args_to_build_date_features, convert_unix_data, \
    save_config_file


if __name__ == '__main__':
    args = get_args_to_build_date_features()
    channel = args.channel
    raw_frequency = args.raw_frequency
    datetime_features = args.datetime_features

    training_data_dir = os.path.join(
        os.environ.get('BIN_NAME'),
        os.environ.get('FORMWATTED_NAME'),
        channel)

    save_config_file(
        folder_dir=training_data_dir,
        args=args)

    ts_train = np.load(os.path.join(training_data_dir, 'ix_train.npy'))
    ts_test = np.load(os.path.join(training_data_dir, 'ix_test.npy'))

    ts_train = convert_unix_data(ts_train, raw_frequency, datetime_features)
    ts_test = convert_unix_data(ts_test, raw_frequency, datetime_features)

    np.save(os.path.join(training_data_dir, 'ts_train.npy'), ts_train)
    np.save(os.path.join(training_data_dir, 'ts_test.npy'), ts_test)
