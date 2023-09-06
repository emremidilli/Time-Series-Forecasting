import os

from settings import TRAINING_DATA_FOLDER, PATCH_SIZE, \
    POOL_SIZE_REDUCTION, POOL_SIZE_TREND, NR_OF_BINS, \
    ARTIFACTS_FOLDER, QUANTILES, NR_OF_FORECAST_PATCHES, MSK_SCALAR

import shutil

import tensorflow as tf

from tsf_model import InputPreProcessor, TargetPreProcessor, FineTuning

from utils import get_fine_tuning_args, read_npy_file, \
    FineTuningCheckpointCallback, get_pre_trained_representation, \
    train_test_split, RamCleaner


if __name__ == '__main__':
    '''
    Fine tunes a given channel.
    A training dataset should be in format of (None, timesteps).
    '''
    args = get_fine_tuning_args()
    print(args)

    channel = input(f'Enter a channel name from {TRAINING_DATA_FOLDER}: \n')
    resume_training = input('Resume training {Y, N}: \n').upper()

    artifacts_dir = os.path.join(ARTIFACTS_FOLDER, channel, 'fine_tune')
    custom_ckpt_dir = os.path.join(artifacts_dir, 'checkpoints', 'ckpt')
    saved_model_dir = os.path.join(artifacts_dir, 'saved_model')
    tensorboard_log_dir = os.path.join(artifacts_dir, 'tboard_logs')
    pre_trained_model_dir = os.path.join(ARTIFACTS_FOLDER,
                                         channel,
                                         'pre_train',
                                         'saved_model')

    lb_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'lb_train.npy'))
    fc_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'fc_train.npy'))
    ts_train = read_npy_file(
        os.path.join(TRAINING_DATA_FOLDER, channel, 'ts_train.npy'))

    input_pre_processor = InputPreProcessor(
        iPatchSize=PATCH_SIZE,
        iPoolSizeReduction=POOL_SIZE_REDUCTION,
        iPoolSizeTrend=POOL_SIZE_TREND,
        iNrOfBins=NR_OF_BINS
    )

    target_pre_processor = TargetPreProcessor(
        iPatchSize=PATCH_SIZE,
        quantiles=QUANTILES)

    dist, tre, sea = input_pre_processor((lb_train, fc_train))
    dist, tre, sea = input_pre_processor.mask_forecast_patches(
        inputs=(dist, tre, sea),
        nr_of_patches=NR_OF_FORECAST_PATCHES,
        msk_scalar=MSK_SCALAR
    )
    qntl = target_pre_processor((lb_train, fc_train))
    ts = input_pre_processor.batch_normalizer(ts_train, training=True)

    ds = tf.data.Dataset.from_tensor_slices(((dist, tre, sea, ts), qntl))
    ds_train = ds
    ds_val = None
    if args.validation_rate > 0:
        ds_train, ds_val = train_test_split(ds, test_rate=args.validation_rate)
        ds_val = ds_val.batch(args.mini_batch_size).prefetch(tf.data.AUTOTUNE)

    ds_train = ds_train.batch(args.mini_batch_size).prefetch(tf.data.AUTOTUNE)

    con_temp_pret = get_pre_trained_representation(pre_trained_model_dir)

    model = FineTuning(
        con_temp_pret=con_temp_pret,
        nr_of_time_steps=NR_OF_FORECAST_PATCHES,
        nr_of_quantiles=len(QUANTILES))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        clipnorm=args.clip_norm)

    checkpoint_callback = FineTuningCheckpointCallback(
        ckpt_dir=custom_ckpt_dir,
        epoch_freq=3)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        write_graph=True,
        write_images=False,
        histogram_freq=1)

    terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()

    ram_cleaner_callback = RamCleaner()

    starting_epoch = 0
    if resume_training == 'Y':
        starting_epoch, _, model, optimizer = checkpoint_callback.\
            get_most_recent_ckpt(model=model, optimizer=optimizer)
    else:
        shutil.rmtree(artifacts_dir, ignore_errors=True)
        os.makedirs(artifacts_dir)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(name='mse'),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.CosineSimilarity(name='cos')
        ]
    )

    print(f'tensorboard --logdir=".{tensorboard_log_dir}" --bind_all')
    model.fit(
        ds_train,
        epochs=args.nr_of_epochs,
        verbose=2,
        validation_data=ds_val,
        initial_epoch=starting_epoch,
        shuffle=False,
        callbacks=[
            terminate_on_nan_callback,
            tensorboard_callback,
            checkpoint_callback,
            ram_cleaner_callback])

    model.save(
        saved_model_dir,
        overwrite=True,
        save_format='tf')

    print('Training completed.')
