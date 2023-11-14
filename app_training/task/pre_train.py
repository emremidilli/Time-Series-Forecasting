import os

import shutil

import tensorflow as tf

from tsf_model import PreTraining

from utils import PreTrainingCheckpointCallback, LearningRateCallback, \
    RamCleaner, get_pre_training_args, get_data_format_config


if __name__ == '__main__':
    '''Pre-trains a model.'''

    args = get_pre_training_args()
    print(args)

    model_id = args.model_id
    resume_training = args.resume_training
    mask_rate = args.mask_rate
    mask_scalar = args.mask_scalar
    mini_batch_size = args.mini_batch_size
    clip_norm = args.clip_norm
    nr_of_encoder_blocks = args.nr_of_encoder_blocks
    nr_of_heads = args.nr_of_heads
    dropout_rate = args.dropout_rate
    encoder_ffn_units = args.encoder_ffn_units
    embedding_dims = args.embedding_dims
    projection_head = args.projection_head
    warmup_steps = args.warmup_steps
    scale_factor = args.scale_factor

    artifacts_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['ARTIFACTS_NAME'],
        model_id,
        'pre_train')
    custom_ckpt_dir = os.path.join(artifacts_dir, 'checkpoints', 'ckpt')
    saved_model_dir = os.path.join(artifacts_dir, 'saved_model')
    tensorboard_log_dir = os.path.join(artifacts_dir, 'tboard_logs')
    dataset_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        model_id,
        'pre_train',
        'dataset')

    config = get_data_format_config(
        folder_path=os.path.join(
            os.environ['BIN_NAME'],
            os.environ['FORMATTED_NAME'],
            model_id))

    ds_train = tf.data.Dataset.load(path=dataset_dir)
    dist, tre, _, _ = next(iter(ds_train))

    ds_train = ds_train.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    checkpoint_callback = PreTrainingCheckpointCallback(
        ckpt_dir=custom_ckpt_dir,
        epoch_freq=25)

    ram_cleaner_callback = RamCleaner()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        write_graph=True,
        write_images=False,
        histogram_freq=1,
        profile_batch='50,70')

    terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()

    mae_optimizer = tf.keras.optimizers.Adam(clipnorm=clip_norm)

    cl_optimizer = tf.keras.optimizers.Adam(clipnorm=clip_norm)

    lookback_coefficient = config['lookback_coefficient']
    nr_of_forecast_patches = int(dist.shape[0] / (lookback_coefficient + 1))
    nr_of_lookback_patches = int(nr_of_forecast_patches * lookback_coefficient)

    starting_epoch = 0
    starting_step = 0
    model = PreTraining(
        iNrOfEncoderBlocks=nr_of_encoder_blocks,
        iNrOfHeads=nr_of_heads,
        fDropoutRate=dropout_rate,
        iEncoderFfnUnits=encoder_ffn_units,
        embedding_dims=embedding_dims,
        iProjectionHeadUnits=projection_head,
        iReducedDims=tre.shape[1],
        fMskRate=mask_rate,
        msk_scalar=mask_scalar,
        iNrOfBins=dist.shape[1],
        iNrOfLookbackPatches=nr_of_lookback_patches,
        iNrOfForecastPatches=nr_of_forecast_patches)
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
        d_model=embedding_dims,
        warmup_steps=warmup_steps,
        scale_factor=scale_factor,
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
            # tensorboard_callback,
            learning_rate_callback,
            checkpoint_callback])

    model.save(
        saved_model_dir,
        overwrite=True,
        save_format='tf')

    print('Training completed.')
