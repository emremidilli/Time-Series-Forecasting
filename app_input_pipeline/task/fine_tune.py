from models import InputPreProcessorFT, TargetPreProcessor

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

    model_id = args.model_id
    patch_size = args.patch_size
    pool_size_reduction = args.pool_size_reduction
    pool_size_trend = args.pool_size_trend
    nr_of_bins = args.nr_of_bins
    mask_scalar = args.mask_scalar
    begin_scalar = args.begin_scalar
    end_scalar = args.end_scalar

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

    nr_of_forecast_patches = int(fc_train.shape[1] / patch_size)

    input_pre_processor = InputPreProcessorFT(
        patch_size=patch_size,
        pool_size_reduction=pool_size_reduction,
        pool_size_trend=pool_size_trend,
        nr_of_bins=nr_of_bins,
        forecast_patches_to_mask=nr_of_forecast_patches,
        mask_scalar=mask_scalar)

    target_pre_processor = TargetPreProcessor(
        patch_size=patch_size,
        begin_scalar=begin_scalar,
        end_scalar=end_scalar)

    input_pre_processor.adapt(inputs=(lb_train, ts_train))
    dist, tre, sea, ts = input_pre_processor(inputs=(lb_train, ts_train))
    lbl, lbl_shifted = target_pre_processor((lb_train, fc_train))

    ds = tf.data.Dataset.from_tensor_slices(
        ((dist, tre, sea, ts, lbl_shifted), lbl))

    sub_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        model_id,
        'fine_tune')

    ds.save(
        os.path.join(sub_dir, 'dataset'))

    tf.saved_model.save(
        obj=input_pre_processor,
        export_dir=os.path.join(sub_dir, 'input_preprocessor'))

    tf.saved_model.save(
        obj=target_pre_processor,
        export_dir=os.path.join(sub_dir, 'target_preprocessor'))
