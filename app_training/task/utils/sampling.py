import numpy as np


def get_random_sample(lb, fc, ts, sampling_ratio):
    '''Picks random elements from the numpy arrays.'''
    np.random.seed(1)
    size_of_sample = int(len(lb) * sampling_ratio)
    all_indices = np.arange(len(lb))
    np.random.shuffle(all_indices)
    rand_indices = all_indices[:size_of_sample]
    rand_indices = np.sort(rand_indices)

    return lb[rand_indices], fc[rand_indices], ts[rand_indices]


def train_test_split(ds, test_rate=0.15):
    '''Splits tf.data.Dataset to train and test datasets.'''
    nr_of_samples = ds.cardinality().numpy()

    train_size = int(nr_of_samples * (1 - test_rate))

    ds_train = ds.take(train_size)
    ds_test = ds.skip(train_size)

    return ds_train, ds_test
