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
