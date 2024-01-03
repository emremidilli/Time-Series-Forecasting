from models import InputPreProcessorPT

import os

import tensorflow as tf

from utils import read_npy_file, get_input_args_pre_training


if __name__ == '__main__':
    '''
    Converts formatted datasets to tf.data.Dataset format
        for pre-training process.
    Saves final dataset and pre-processor.
    '''
    args = get_input_args_pre_training()
    print(args)

    model_id = args.model_id
    patch_size = args.patch_size
    pool_size_trend = args.pool_size_trend

    training_data_folder = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['FORMATTED_NAME'])

    lb_train = read_npy_file(
        os.path.join(training_data_folder, model_id, 'lb_train.npy'),
        dtype='float32')
    fc_train = read_npy_file(
        os.path.join(training_data_folder, model_id, 'fc_train.npy'),
        dtype='float32')
    ts_train = read_npy_file(
        os.path.join(training_data_folder, model_id, 'ts_train.npy'),
        dtype='int32')

    nr_of_covariates = lb_train.shape[-1]
    input_pre_processor = InputPreProcessorPT(
        patch_size=patch_size,
        pool_size_trend=pool_size_trend,
        nr_of_covariates=nr_of_covariates)

    input_pre_processor.adapt((lb_train, fc_train, ts_train))
    tre, sea, res, ts = input_pre_processor((lb_train, fc_train, ts_train))

    ds_train = tf.data.Dataset.from_tensor_slices(
        (tre, sea, res, ts))

    sub_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        model_id)

    ds_train.save(
        os.path.join(sub_dir, 'dataset'))

    input_pre_processor.save(
        os.path.join(sub_dir, 'input_preprocessor'),
        overwrite=True,
        save_format='tf')
