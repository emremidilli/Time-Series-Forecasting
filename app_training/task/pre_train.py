import mlflow

import os

import shutil

import tensorflow as tf

from tsf_model import PreTraining

from utils import PreTrainingCheckpointCallback, LearningRateCallback, \
    RamCleaner, get_pre_training_args, get_data_format_config, \
    train_test_split


if __name__ == '__main__':
    '''
    Pre-trains a foundation model.
    Each training job is logged to databricks with mlflow.
    This job can be interrupted by the user.
    After interruption, it can be re-run and continue to training
        from where it left.
    '''

    args = get_pre_training_args()

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
    validation_rate = args.validation_rate
    mae_threshold_comp = args.mae_threshold_comp
    mae_threshold_tre = args.mae_threshold_tre
    mae_threshold_sea = args.mae_threshold_sea
    cl_threshold = args.cl_threshold
    cl_margin = args.cl_margin
    save_model = args.save_model
    patch_size = args.patch_size

    artifacts_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['ARTIFACTS_NAME'],
        model_id)

    custom_ckpt_dir = os.path.join(artifacts_dir, 'checkpoints', 'ckpt')
    tensorboard_log_dir = os.path.join(artifacts_dir, 'tboard_logs')
    csv_logs_dir = os.path.join(artifacts_dir, 'csv_logs', 'training.log')
    dataset_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        model_id,
        'dataset')

    input_pipeline_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        model_id,
        'input_preprocessor')

    config = get_data_format_config(
        folder_path=os.path.join(
            os.environ['BIN_NAME'],
            os.environ['FORMATTED_NAME'],
            model_id))

    ds = tf.data.Dataset.load(path=dataset_dir)
    tre, _, _, _ = next(iter(ds))

    ds_train = ds
    ds_val = None
    if validation_rate > 0:
        ds_train, ds_val = train_test_split(ds, test_rate=validation_rate)
        ds_val = ds_val.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

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

    csv_logger_callback = tf.keras.callbacks.CSVLogger(
        filename=csv_logs_dir,
        separator=";",
        append=True)

    terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()

    mae_comp_optimizer = tf.keras.optimizers.Adam(clipnorm=clip_norm)
    mae_tre_optimizer = tf.keras.optimizers.Adam(clipnorm=clip_norm)
    mae_sea_optimizer = tf.keras.optimizers.Adam(clipnorm=clip_norm)

    cl_optimizer = tf.keras.optimizers.Adam(clipnorm=clip_norm)

    lookback_coefficient = config['lookback_coefficient']
    nr_of_forecast_patches = int(tre.shape[0] / (lookback_coefficient + 1))
    nr_of_forecast_patches = int(nr_of_forecast_patches / patch_size)
    nr_of_lookback_patches = int(nr_of_forecast_patches * lookback_coefficient)

    pre_processor = tf.keras.models.load_model(input_pipeline_dir)

    starting_epoch = 0
    starting_step = 0
    model = PreTraining(
        nr_of_covariates=tre.shape[-1],
        patch_size=patch_size,
        nr_of_encoder_blocks=nr_of_encoder_blocks,
        nr_of_heads=nr_of_heads,
        dropout_rate=dropout_rate,
        encoder_ffn_units=encoder_ffn_units,
        embedding_dims=embedding_dims,
        projection_head_units=projection_head,
        msk_rate=mask_rate,
        msk_scalar=mask_scalar,
        nr_of_lookback_patches=nr_of_lookback_patches,
        nr_of_forecast_patches=nr_of_forecast_patches,
        mae_threshold_comp=mae_threshold_comp,
        mae_threshold_tre=mae_threshold_tre,
        mae_threshold_sea=mae_threshold_sea,
        cl_threshold=cl_threshold,
        cl_margin=cl_margin,
        pre_processor=pre_processor)
    if resume_training == 'Y':
        starting_epoch,
        starting_step,
        model,
        mae_comp_optimizer,
        mae_tre_optimizer,
        mae_sea_optimizer,
        cl_optimizer = \
            checkpoint_callback.get_most_recent_ckpt(
                model,
                mae_comp_optimizer,
                mae_tre_optimizer,
                mae_sea_optimizer,
                cl_optimizer)
    else:
        shutil.rmtree(artifacts_dir, ignore_errors=True)
        os.makedirs(artifacts_dir)

        os.makedirs(os.path.dirname(csv_logs_dir))

    model.compile(
        mae_comp_optimizer=mae_comp_optimizer,
        mae_tre_optimizer=mae_tre_optimizer,
        mae_sea_optimizer=mae_sea_optimizer,
        cl_optimizer=cl_optimizer)

    learning_rate_callback = LearningRateCallback(
        d_model=embedding_dims,
        warmup_steps=warmup_steps,
        scale_factor=scale_factor,
        remained_step_nr=starting_step)

    print(f'tensorboard --logdir=".{tensorboard_log_dir}" --bind_all')
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.nr_of_epochs,
        verbose=2,
        initial_epoch=starting_epoch,
        shuffle=False,
        callbacks=[
            terminate_on_nan_callback,
            ram_cleaner_callback,
            # tensorboard_callback,
            # csv_logger_callback,
            learning_rate_callback,
            checkpoint_callback])

    mlflow.login()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(f'/{model_id}')

    with mlflow.start_run():
        history_logs = history.history
        mlflow.log_params(vars(args))
        mlflow.log_table(
            data=history_logs,
            artifact_file="history_logs.json")

        for metric in list(history_logs.keys()):
            mlflow.log_metric(metric, history_logs[metric][-1])

        if save_model == "Y":
            mlflow.tensorflow.log_model(
                model,
                artifact_path='saved_model')

    print('Training completed.')
