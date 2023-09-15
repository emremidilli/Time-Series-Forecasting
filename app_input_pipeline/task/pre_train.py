import os

from settings import TRAINING_DATA_FOLDER, PREPROCESSING_DIR

import tensorflow as tf

from models import InputPreProcessor

from utils import get_random_sample, read_npy_file, get_input_args_pre_training


if __name__ == '__main__':
    '''
    Converts formatted datasets to tf.data.Dataset format
        for pre-training process.
    Saves final dataset and pre-processor.
    '''
    args = get_input_args_pre_training()
    print(args)

    channel = args.channel
    patch_size = args.patch_size
    pool_size_reduction = args.pool_size_reduction
    pool_size_trend = args.pool_size_trend
    nr_of_bins = args.nr_of_bins
    pre_train_ratio = args.pre_train_ratio

    lb_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'lb_train.npy'),
        dtype='float32')
    fc_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'fc_train.npy'),
        dtype='float32')
    ts_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'ts_train.npy'),
        dtype='int32')

    lb_train, fc_train, ts_train = get_random_sample(
        lb=lb_train,
        fc=fc_train,
        ts=ts_train,
        sampling_ratio=pre_train_ratio)

    input_pre_processor = InputPreProcessor(
        iPatchSize=patch_size,
        iPoolSizeReduction=pool_size_reduction,
        iPoolSizeTrend=pool_size_trend,
        iNrOfBins=nr_of_bins)

    dist, tre, sea = input_pre_processor((lb_train, fc_train))
    ts_train = input_pre_processor.batch_normalizer(ts_train, training=True)

    ds_train = tf.data.Dataset.from_tensor_slices(
        (dist, tre, sea, ts_train))

    sub_dir = os.path.join(PREPROCESSING_DIR, channel, 'pre_train')

    ds_train.save(
        os.path.join(sub_dir, 'dataset'))

    input_pre_processor.save(
        os.path.join(sub_dir, 'input_preprocessor'),
        overwrite=True,
        save_format='tf')
