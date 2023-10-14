from models import InputPreProcessorPT

import os

import tensorflow as tf

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

    training_data_folder = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['FORMATTED_NAME'])

    lb_train = read_npy_file(
        os.path.join(training_data_folder, channel, 'lb_train.npy'),
        dtype='float32')
    fc_train = read_npy_file(
        os.path.join(training_data_folder, channel, 'fc_train.npy'),
        dtype='float32')
    ts_train = read_npy_file(
        os.path.join(training_data_folder, channel, 'ts_train.npy'),
        dtype='int32')

    input_pre_processor = InputPreProcessorPT(
        patch_size=patch_size,
        pool_size_reduction=pool_size_reduction,
        pool_size_trend=pool_size_trend,
        nr_of_bins=nr_of_bins)

    input_pre_processor.adapt((lb_train, fc_train, ts_train))
    dist, tre, sea, ts = input_pre_processor((lb_train, fc_train, ts_train))

    dist, tre, sea, ts = get_random_sample(
        dist=dist,
        tre=tre,
        sea=sea,
        ts=ts,
        sampling_ratio=pre_train_ratio)

    ds_train = tf.data.Dataset.from_tensor_slices(
        (dist, tre, sea, ts))

    sub_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        channel,
        'pre_train')

    ds_train.save(
        os.path.join(sub_dir, 'dataset'))

    tf.saved_model.save(
        obj=input_pre_processor,
        export_dir=os.path.join(sub_dir, 'input_preprocessor'))
