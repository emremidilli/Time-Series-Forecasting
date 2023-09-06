'''
Trains a pre-training model for a univariate forecasting model.
Pre-training is done on a small portion of the training dataset.
A pre-training ratio is used to select the pre-training dataset
    from the training dataset.
'''

import os

from settings import TRAINING_DATA_FOLDER, PATCH_SIZE, \
    POOL_SIZE_REDUCTION, POOL_SIZE_TREND, NR_OF_BINS, \
    MASK_RATE, MSK_SCALAR, ARTIFACTS_FOLDER, \
    NR_OF_LOOKBACK_PATCHES, NR_OF_FORECAST_PATCHES

import shutil

import tensorflow as tf

from tsf_model import InputPreProcessor, PreTraining

from utils import PreTrainingCheckpointCallback, LearningRateCallback, \
    get_random_sample, RamCleaner, get_pre_training_args, read_npy_file


if __name__ == '__main__':
    '''
    Pre-trains a given channel.
    A training dataset should be in format of (None, timesteps).
    '''
    args = get_pre_training_args()
    print(args)

    channel = input(f'Enter a channel name from {TRAINING_DATA_FOLDER}: \n')
    resume_training = input('Resume training {Y, N}: \n').upper()

    artifacts_dir = os.path.join(ARTIFACTS_FOLDER, channel, 'pre_train')
    custom_ckpt_dir = os.path.join(artifacts_dir, 'checkpoints', 'ckpt')
    saved_model_dir = os.path.join(artifacts_dir, 'saved_model')
    tensorboard_log_dir = os.path.join(artifacts_dir, 'tboard_logs')

    lb_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'lb_train.npy'))
    fc_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'fc_train.npy'))
    ts_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'ts_train.npy'))

    lb_train, fc_train, ts_train = get_random_sample(
        lb=lb_train,
        fc=fc_train,
        ts=ts_train,
        sampling_ratio=args.pre_train_ratio)

    input_pre_processor = InputPreProcessor(
        iPatchSize=PATCH_SIZE,
        iPoolSizeReduction=POOL_SIZE_REDUCTION,
        iPoolSizeTrend=POOL_SIZE_TREND,
        iNrOfBins=NR_OF_BINS
    )
    dist, tre, sea = input_pre_processor((lb_train, fc_train))
    ts_train = input_pre_processor.batch_normalizer(ts_train, training=True)

    ds_train = tf.data.Dataset.from_tensor_slices(
        (dist, tre, sea, ts_train)).batch(
            args.mini_batch_size).prefetch(tf.data.AUTOTUNE)

    checkpoint_callback = PreTrainingCheckpointCallback(
        ckpt_dir=custom_ckpt_dir,
        epoch_freq=3)

    ram_cleaner_callback = RamCleaner()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        write_graph=True,
        write_images=False,
        histogram_freq=1)

    terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()

    mae_optimizer = tf.keras.optimizers.Adam(clipnorm=args.clip_norm)

    cl_optimizer = tf.keras.optimizers.Adam(clipnorm=args.clip_norm)

    starting_epoch = 0
    starting_step = 0
    model = PreTraining(
        iNrOfEncoderBlocks=args.nr_of_encoder_blocks,
        iNrOfHeads=args.nr_of_heads,
        fDropoutRate=args.dropout_rate,
        iEncoderFfnUnits=args.encoder_ffn_units,
        embedding_dims=args.embedding_dims,
        iProjectionHeadUnits=args.projection_head,
        iReducedDims=tre.shape[2],
        fMskRate=MASK_RATE,
        msk_scalar=MSK_SCALAR,
        iNrOfBins=NR_OF_BINS,
        iNrOfLookbackPatches=NR_OF_LOOKBACK_PATCHES,
        iNrOfForecastPatches=NR_OF_FORECAST_PATCHES)
    if resume_training == 'Y':
        starting_epoch, starting_step, model, mae_optimizer, cl_optimizer = \
            checkpoint_callback.get_most_recent_ckpt(
                model,
                mae_optimizer,
                cl_optimizer)
    else:
        shutil.rmtree(artifacts_dir, ignore_errors=True)
        os.makedirs(artifacts_dir)

    model.compile(
        mae_optimizer=mae_optimizer,
        cl_optimizer=cl_optimizer)

    learning_rate_callback = LearningRateCallback(
        d_model=args.embedding_dims,
        remained_step_nr=starting_step)

    print(f'tensorboard --logdir=".{tensorboard_log_dir}" --bind_all')
    history = model.fit(
        ds_train,
        epochs=args.nr_of_epochs,
        verbose=2,
        initial_epoch=starting_epoch,
        shuffle=False,
        callbacks=[
            terminate_on_nan_callback,
            ram_cleaner_callback,
            tensorboard_callback,
            learning_rate_callback,
            checkpoint_callback])

    model.save(
        saved_model_dir,
        overwrite=True,
        save_format='tf')

    print('Training completed.')
