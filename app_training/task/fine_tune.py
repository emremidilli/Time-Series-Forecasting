import os

import shutil

import tensorflow as tf

from tsf_model import FineTuning

from utils import get_fine_tuning_args, FineTuningCheckpointCallback, \
    upload_model, load_model, train_test_split, RamCleaner, log_experiments


if __name__ == '__main__':
    '''Fine tunes a pre-trained model.'''
    args = get_fine_tuning_args()
    print(args)

    model_id = args.model_id
    pre_trained_model_id = args.pre_trained_model_id
    dataset_id = args.dataset_id
    resume_training = args.resume_training
    validation_rate = args.validation_rate
    mini_batch_size = args.mini_batch_size
    learning_rate = args.learning_rate
    clip_norm = args.clip_norm
    nr_of_epochs = args.nr_of_epochs

    artifacts_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['ARTIFACTS_NAME'],
        model_id)
    custom_ckpt_dir = os.path.join(artifacts_dir, 'checkpoints', 'ckpt')

    dataset_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        dataset_id)

    pre_trained_model = load_model(model_id=pre_trained_model_id)

    ds = tf.data.Dataset.load(path=os.path.join(dataset_dir, 'dataset_train'))

    (_, _, _, _), lbl = next(iter(ds))

    ds_train = ds
    ds_val = None
    if validation_rate > 0:
        ds_train, ds_val = train_test_split(ds, test_rate=validation_rate)
        ds_val = ds_val.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    ds_train = ds_train.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    ds_test = tf.data.Dataset.load(
        path=os.path.join(dataset_dir, 'dataset_test'))
    ds_test = ds_test.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    model = FineTuning(
        revIn_tre=pre_trained_model.revIn_tre,
        revIn_sea=pre_trained_model.revIn_sea,
        revIn_res=pre_trained_model.revIn_res,
        patch_tokenizer=pre_trained_model.patch_tokenizer,
        tre_embedding=pre_trained_model.tre_embedding,
        sea_embedding=pre_trained_model.sea_embedding,
        res_embedding=pre_trained_model.res_embedding,
        encoder_representation=pre_trained_model.encoder_representation,
        nr_of_timesteps=lbl.shape[0],
        nr_of_covariates=lbl.shape[-1],
        shared_prompt=pre_trained_model.shared_prompt,
        decoder_tre=pre_trained_model.decoder_tre,
        decoder_sea=pre_trained_model.decoder_sea,
        decoder_res=pre_trained_model.decoder_res)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clip_norm)

    checkpoint_callback = FineTuningCheckpointCallback(
        ckpt_dir=custom_ckpt_dir,
        epoch_freq=10)

    terminate_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()

    ram_cleaner_callback = RamCleaner()

    metric_to_monitor = 'mae'
    if validation_rate > 0:
        metric_to_monitor = 'val_mae'
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=metric_to_monitor,
        patience=10,
        start_from_epoch=50,
        restore_best_weights=True)

    starting_epoch = 0
    if resume_training == 'Y':
        starting_epoch, _, model, optimizer = checkpoint_callback.\
            get_most_recent_ckpt(model=model, optimizer=optimizer)
    else:
        shutil.rmtree(artifacts_dir, ignore_errors=True)
        os.makedirs(artifacts_dir)

    model.compile(
        run_eagerly=False,
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(name='mse'),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.CosineSimilarity(name='cos')
        ])

    callbacks = [
        terminate_on_nan_callback,
        checkpoint_callback,
        ram_cleaner_callback,
        early_stopping]

    history = model.fit(
        ds_train,
        epochs=nr_of_epochs,
        verbose=2,
        validation_data=ds_val,
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
        parameters=vars(args),
        model_type='ft')

    upload_model(model=model, model_id=model_id)

    print('Training completed.')
