import os

from settings import TRAINING_DATA_FOLDER, PREPROCESSING_DIR

import tensorflow as tf

from models import InputPreProcessor, TargetPreProcessor

from utils import read_npy_file, get_input_args_fine_tuning


if __name__ == '__main__':
    '''
    Converts formatted datasets to tf.data.Dataset format
        for fine-tuning process.
    Saves final dataset and pre-processor.
    '''
    args = get_input_args_fine_tuning()
    print(args)

    channel = args.channel
    patch_size = args.patch_size
    pool_size_reduction = args.pool_size_reduction
    pool_size_trend = args.pool_size_trend
    nr_of_bins = args.nr_of_bins
    quantiles = args.quantiles
    mask_scalar = args.mask_scalar

    lb_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'lb_train.npy'),
        dtype='float32')
    fc_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'fc_train.npy'),
        dtype='float32')
    ts_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'ts_train.npy'),
        dtype='int32')

    nr_of_forecast_patches = int(fc_train.shape[1] / patch_size)

    input_pre_processor = InputPreProcessor(
        iPatchSize=patch_size,
        iPoolSizeReduction=pool_size_reduction,
        iPoolSizeTrend=pool_size_trend,
        iNrOfBins=nr_of_bins)

    target_pre_processor = TargetPreProcessor(
        iPatchSize=patch_size,
        quantiles=quantiles)

    dist, tre, sea = input_pre_processor((lb_train, fc_train))
    dist, tre, sea = input_pre_processor.mask_forecast_patches(
        inputs=(dist, tre, sea),
        nr_of_patches=nr_of_forecast_patches,
        msk_scalar=mask_scalar
    )
    qntl = target_pre_processor((lb_train, fc_train))
    ts = input_pre_processor.batch_normalizer(ts_train, training=True)

    ds = tf.data.Dataset.from_tensor_slices(((dist, tre, sea, ts), qntl))

    sub_dir = os.path.join(PREPROCESSING_DIR, channel, 'fine_tune')

    ds.save(
        os.path.join(sub_dir, 'dataset'))

    input_pre_processor.save(
        os.path.join(sub_dir, 'input_preprocessor'),
        overwrite=True,
        save_format='tf')

    target_pre_processor.save(
        os.path.join(sub_dir, 'target_preprocessor'),
        overwrite=True,
        save_format='tf')
