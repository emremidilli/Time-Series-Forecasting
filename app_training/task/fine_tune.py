import os

import shutil

import tensorflow as tf

from tsf_model import FineTuning

from utils import get_fine_tuning_args, FineTuningCheckpointCallback, \
    get_pre_trained_representation, train_test_split, RamCleaner, \
    get_data_format_config


if __name__ == '__main__':
    '''Fine tunes a given channel.'''
    args = get_fine_tuning_args()
    print(args)

    channel = args.channel
    resume_training = args.resume_training
    validation_rate = args.validation_rate
    mini_batch_size = args.mini_batch_size
    learning_rate = args.learning_rate
    clip_norm = args.clip_norm
    nr_of_epochs = args.nr_of_epochs

    artifacts_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['ARTIFACTS_NAME'],
        channel,
        'fine_tune')
    custom_ckpt_dir = os.path.join(artifacts_dir, 'checkpoints', 'ckpt')
    saved_model_dir = os.path.join(artifacts_dir, 'saved_model')
    tensorboard_log_dir = os.path.join(artifacts_dir, 'tboard_logs')
    pre_trained_model_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['ARTIFACTS_NAME'],
        channel,
        'pre_train',
        'saved_model')
    dataset_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        channel,
        'fine_tune',
        'dataset')

    config = get_data_format_config(
        folder_path=os.path.join(
            os.environ['BIN_NAME'],
            os.environ['FORMWATTED_NAME'],
            channel))

    ds = tf.data.Dataset.load(path=dataset_dir)

    (dist, _, _, _), qtl = next(iter(ds))
    lookback_coefficient = config['lookback_coefficient']
    nr_of_forecast_patches = int(dist.shape[0] / (lookback_coefficient + 1))

    ds_train = ds
    ds_val = None
    if validation_rate > 0:
        ds_train, ds_val = train_test_split(ds, test_rate=validation_rate)
        ds_val = ds_val.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    ds_train = ds_train.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    con_temp_pret = get_pre_trained_representation(pre_trained_model_dir)

    model = FineTuning(
        con_temp_pret=con_temp_pret,
        nr_of_time_steps=nr_of_forecast_patches,
        nr_of_quantiles=qtl.shape[1])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clip_norm)

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
        epochs=nr_of_epochs,
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
