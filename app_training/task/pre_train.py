import os

import shutil

import tensorflow as tf

from tsf_model import PreTraining

from utils import PreTrainingCheckpointCallback, LearningRateCallback, \
    RamCleaner, get_pre_training_args, \
    upload_model, log_experiments, MaskingCallback


if __name__ == '__main__':
    '''
    Pre-trains a foundation model.
    Each training job is logged to databricks with mlflow.
    Trained model is saved to AWS S3 bucket.
    This job can be interrupted by the user.
    After interruption, it can be re-run and continue to training
        from where it left.
    '''

    args = get_pre_training_args()

    model_id = args.model_id
    dataset_id = args.dataset_id
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
    mae_threshold_comp = args.mae_threshold_comp
    mae_threshold_tre = args.mae_threshold_tre
    mae_threshold_sea = args.mae_threshold_sea
    cl_threshold = args.cl_threshold
    cl_margin = args.cl_margin
    save_model = args.save_model
    patch_size = args.patch_size
    lookback_coefficient = args.lookback_coefficient
    prompt_pool_size = args.prompt_pool_size
    nr_of_most_similar_prompts = args.nr_of_most_similar_prompts
    concat_train_val = args.concat_train_val

    artifacts_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['ARTIFACTS_NAME'],
        model_id)

    custom_ckpt_dir = os.path.join(artifacts_dir, 'checkpoints', 'ckpt')
    dataset_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        dataset_id)

    input_pipeline_dir = os.path.join(dataset_dir, 'input_preprocessor')

    ds_train = tf.data.Dataset.load(
        path=os.path.join(dataset_dir, 'dataset_train'))

    ds_val = tf.data.Dataset.load(
        path=os.path.join(dataset_dir, 'dataset_validation'))

    if concat_train_val == 'Y':
        ds_train = ds_train.concatenate(ds_val)
        ds_val = None

    ds_test = tf.data.Dataset.load(
        path=os.path.join(dataset_dir, 'dataset_test'))

    tre, _, _, _ = next(iter(ds_train))

    ds_train = ds_train.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    if ds_val is not None:
        ds_val = ds_val.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    checkpoint_callback = PreTrainingCheckpointCallback(
        ckpt_dir=custom_ckpt_dir,
        epoch_freq=25)

    mae_comp_optimizer = tf.keras.optimizers.Adam(clipnorm=clip_norm)
    mae_tre_optimizer = tf.keras.optimizers.Adam(clipnorm=clip_norm)
    mae_sea_optimizer = tf.keras.optimizers.Adam(clipnorm=clip_norm)

    cl_optimizer = tf.keras.optimizers.Adam(clipnorm=clip_norm)

    contrastive_learning_patches = \
        int(tre.shape[0] / (lookback_coefficient + 1))
    contrastive_learning_patches = \
        int(contrastive_learning_patches / patch_size)

    masked_auto_encoder_patches = int(tre.shape[0] / patch_size)

    nr_of_timesteps = tre.shape[0]

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
        msk_scalar=mask_scalar,
        nr_of_timesteps=nr_of_timesteps,
        contrastive_learning_patches=contrastive_learning_patches,
        mae_threshold_comp=mae_threshold_comp,
        mae_threshold_tre=mae_threshold_tre,
        mae_threshold_sea=mae_threshold_sea,
        cl_threshold=cl_threshold,
        cl_margin=cl_margin,
        pre_processor=pre_processor,
        prompt_pool_size=prompt_pool_size,
        nr_of_most_similar_prompts=nr_of_most_similar_prompts)

    if resume_training == 'Y':
        starting_epoch, \
            starting_step, \
            model, \
            mae_comp_optimizer, \
            mae_tre_optimizer, \
            mae_sea_optimizer, \
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

    masking_callback = MaskingCallback(
        nr_of_patches=masked_auto_encoder_patches,
        masking_rate=mask_rate,
        masks_feature='masks')

    masking_callback_cl = MaskingCallback(
        nr_of_patches=(
            masked_auto_encoder_patches - contrastive_learning_patches),
        masking_rate=mask_rate,
        masks_feature='cl_masks')

    ram_cleaner_callback = RamCleaner()

    terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()

    metric_to_monitor = 'mae_comp'
    if ds_val is not None:
        metric_to_monitor = 'val_mae_comp'
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=metric_to_monitor,
        patience=10,
        start_from_epoch=50,
        restore_best_weights=True)

    callbacks = [
        terminate_on_nan_callback,
        ram_cleaner_callback,
        learning_rate_callback,
        checkpoint_callback,
        masking_callback,
        masking_callback_cl]

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.nr_of_epochs,
        verbose=2,
        initial_epoch=starting_epoch,
        shuffle=False,
        callbacks=callbacks)

    log_experiments(
        model_id=model_id,
        model=model,
        history=history,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        parameters=vars(args))

    if save_model == "Y":
        upload_model(model=model, model_id=model_id)

    print('Training completed.')
