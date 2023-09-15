def train_test_split(ds, test_rate=0.15):
    '''Splits tf.data.Dataset to train and test datasets.'''
    nr_of_samples = ds.cardinality().numpy()

    train_size = int(nr_of_samples * (1 - test_rate))

    ds_train = ds.take(train_size)
    ds_test = ds.skip(train_size)

    return ds_train, ds_test
