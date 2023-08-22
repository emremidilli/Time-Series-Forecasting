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


def train_test_split(*arrays, test_rate=0.15):
    '''
    Applies train-test split to numpy arrays.
    *arrays: list of arrays
    test_rate: float between 0 and 1.

    returns: tuple consists ofsplitted arrays in
        (train_1, test_1, train_2, test_2 ...) format.
    '''
    np.random.seed(1)

    size_of_test = int(len(arrays[0]) * test_rate)

    all_indices = np.arange(len(arrays[0]))
    np.random.shuffle(all_indices)

    train_indices = all_indices[size_of_test:]
    train_indices = np.sort(train_indices)

    test_indices = all_indices[:size_of_test]
    test_indices = np.sort(test_indices)

    return_arrays = []

    for arr in arrays:
        return_arrays.append(np.take(arr, train_indices, axis=0).shape)
        return_arrays.append(np.take(arr, test_indices, axis=0).shape)

    return tuple(return_arrays)
