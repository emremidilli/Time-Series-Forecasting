import numpy as np

import tensorflow as tf


def get_random_sample(tre, sea, res, ts, sampling_ratio):
    '''Picks random elements from tf.tensors.'''
    np.random.seed(1)
    size_of_sample = int(len(tre) * sampling_ratio)
    all_indices = np.arange(len(tre))
    np.random.shuffle(all_indices)
    rand_indices = all_indices[:size_of_sample]
    rand_indices = np.sort(rand_indices)

    return tf.gather(tre, rand_indices), \
        tf.gather(sea, rand_indices), \
        tf.gather(res, rand_indices), \
        tf.gather(ts, rand_indices)


def train_test_split(ds, test_rate=0.15):
    '''Splits tf.data.Dataset to train and test datasets.'''
    nr_of_samples = ds.cardinality().numpy()

    train_size = int(nr_of_samples * (1 - test_rate))

    ds_train = ds.take(train_size)
    ds_test = ds.skip(train_size)

    return ds_train, ds_test
