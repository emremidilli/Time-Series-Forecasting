from models import InputPreProcessor, TargetPreProcessor

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

    begin_scalar = args.begin_scalar
    end_scalar = args.end_scalar

    training_data_folder = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['FORMATTED_NAME'],
        input_dataset_id)

    lb_train = read_npy_file(
        os.path.join(training_data_folder, 'lb_train.npy'),
        dtype='float32')
    fc_train = read_npy_file(
        os.path.join(training_data_folder, 'fc_train.npy'),
        dtype='float32')
    ts_train = read_npy_file(
        os.path.join(training_data_folder, 'ts_train.npy'),
        dtype='int32')

    nr_of_covariates = lb_train.shape[-1]
    input_pre_processor = InputPreProcessorFT(
        pool_size_trend=pool_size_trend,
        nr_of_covariates=nr_of_covariates,
        sigma=sigma,
        scale_data=scale_data)

    target_pre_processor = TargetPreProcessor(
        begin_scalar=begin_scalar,
        end_scalar=end_scalar)

    input_pre_processor.adapt(inputs=(lb_train, ts_train))
    lb_tre, lb_sea, lb_res, ts = input_pre_processor(
        inputs=(lb_train, ts_train))
    lbl, lbl_shifted = target_pre_processor((fc_train))

    ds = tf.data.Dataset.from_tensor_slices(
        ((lb_tre, lb_sea, lb_res, ts, lbl_shifted), lbl))

    sub_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        output_dataset_id)

    ds.save(
        os.path.join(sub_dir, 'dataset'))

    tf.saved_model.save(
        obj=input_pre_processor,
        export_dir=os.path.join(sub_dir, 'input_preprocessor'))

    tf.saved_model.save(
        obj=target_pre_processor,
        export_dir=os.path.join(sub_dir, 'target_preprocessor'))
