from models import InputPreProcessor, TargetPreProcessor

import numpy as np

import os

import tensorflow as tf

from utils import read_npy_file, get_input_args_fine_tuning


if __name__ == '__main__':
    '''
    Converts formatted datasets to tf.data.Dataset format
        for fine-tuning process.
    Saves final dataset and pre-processor.
    '''
    args = get_input_args_fine_tuning()
    print(args)
    input_dataset_id = args.input_dataset_id
    output_dataset_id = args.output_dataset_id
    pool_size_trend = args.pool_size_trend
    sigma = args.sigma
    scale_data = args.scale_data
    if scale_data.upper().strip() == 'Y':
        scale_data = True
    else:
        scale_data = False

    dataset_folder = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['FORMATTED_NAME'],
        input_dataset_id)

    lb_train = read_npy_file(
        os.path.join(dataset_folder, 'lb_train.npy'),
        dtype='float32')

    fc_train = read_npy_file(
        os.path.join(dataset_folder, 'fc_train.npy'),
        dtype='float32')

    ts_train = read_npy_file(
        os.path.join(dataset_folder, 'ts_train.npy'),
        dtype='int32')

    lb_val = read_npy_file(
        os.path.join(dataset_folder, 'lb_val.npy'),
        dtype='float32')

    fc_val = read_npy_file(
        os.path.join(dataset_folder, 'fc_val.npy'),
        dtype='float32')

    ts_val = read_npy_file(
        os.path.join(dataset_folder, 'ts_val.npy'),
        dtype='int32')

    lb_test = read_npy_file(
        os.path.join(dataset_folder, 'lb_test.npy'),
        dtype='float32')

    fc_test = read_npy_file(
        os.path.join(dataset_folder, 'fc_test.npy'),
        dtype='float32')

    ts_test = read_npy_file(
        os.path.join(dataset_folder, 'ts_test.npy'),
        dtype='int32')

    nr_of_covariates = lb_train.shape[-1]
    input_pre_processor = InputPreProcessor(
        pool_size_trend=pool_size_trend,
        nr_of_covariates=nr_of_covariates,
        sigma=sigma,
        scale_data=scale_data)

    lb_train_val = np.concatenate((lb_train, lb_val), axis=0)
    ts_train_val = np.concatenate((ts_train, ts_val), axis=0)

    input_pre_processor.adapt((lb_train_val, ts_train_val))
    lb_tre_train, lb_sea_train, lb_res_train, ts_train = \
        input_pre_processor((lb_train, ts_train))
    lb_tre_val, lb_sea_val, lb_res_val, ts_val = \
        input_pre_processor((lb_val, ts_val))
    lb_tre_test, lb_sea_test, lb_res_test, ts_test = \
        input_pre_processor((lb_test, ts_test))

    target_pre_processor = TargetPreProcessor(scale_data=scale_data)
    lbl_train = target_pre_processor((fc_train))
    lbl_val = target_pre_processor((fc_val))
    lbl_test = target_pre_processor((fc_test))

    ds_train = tf.data.Dataset.from_tensor_slices(
        ((lb_tre_train, lb_sea_train, lb_res_train, ts_train), lbl_train))

    ds_val = tf.data.Dataset.from_tensor_slices(
        ((lb_tre_val, lb_sea_val, lb_res_val, ts_val), lbl_val))

    ds_test = tf.data.Dataset.from_tensor_slices(
        ((lb_tre_test, lb_sea_test, lb_res_test, ts_test), lbl_test))

    sub_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        output_dataset_id)

    ds_train.save(
        os.path.join(sub_dir, 'dataset_train'))

    ds_val.save(
        os.path.join(sub_dir, 'dataset_validation'))

    ds_test.save(
        os.path.join(sub_dir, 'dataset_test'))

    tf.saved_model.save(
        obj=input_pre_processor,
        export_dir=os.path.join(sub_dir, 'input_preprocessor'))

    tf.saved_model.save(
        obj=target_pre_processor,
        export_dir=os.path.join(sub_dir, 'target_preprocessor'))
