import os

import shutil

import tensorflow as tf

from tsf_model import FineTuning

from utils import get_fine_tuning_args, FineTuningCheckpointCallback, \
    upload_model, load_model, train_test_split, RamCleaner


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
    alpha_regulizer = args.alpha_regulizer
    l1_ratio = args.l1_ratio
    nr_of_layers = args.nr_of_layers
    hidden_dims = args.hidden_dims
    nr_of_heads = args.nr_of_heads
    dropout_rate = args.dropout_rate
    pre_trained_lookback_coefficient = args.pre_trained_lookback_coefficient

    artifacts_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['ARTIFACTS_NAME'],
        model_id)
    custom_ckpt_dir = os.path.join(artifacts_dir, 'checkpoints', 'ckpt')

    dataset_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        dataset_id,
        'dataset')

    pre_trained_model = load_model(model_id=pre_trained_model_id)

    ds = tf.data.Dataset.load(path=dataset_dir)

    ds_train = ds
    ds_val = None
    if validation_rate > 0:
        ds_train, ds_val = train_test_split(ds, test_rate=validation_rate)
        ds_val = ds_val.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    ds_train = ds_train.batch(mini_batch_size).prefetch(tf.data.AUTOTUNE)

    model = FineTuning(
        num_layers=nr_of_layers,
        hidden_dims=hidden_dims,
        nr_of_heads=nr_of_heads,
        dff=hidden_dims,
        dropout_rate=dropout_rate,
        pre_trained_lookback_coefficient=pre_trained_lookback_coefficient,
        msk_scalar=pre_trained_model.patch_masker.get_config()['msk_scalar'],
        revIn_tre=pre_trained_model.revIn_tre,
        revIn_sea=pre_trained_model.revIn_sea,
        revIn_res=pre_trained_model.revIn_res,
        patch_tokenizer=pre_trained_model.patch_tokenizer,
        encoder_representation=pre_trained_model.encoder_representation,
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

    starting_epoch = 0
    if resume_training == 'Y':
        starting_epoch, _, model, optimizer = checkpoint_callback.\
            get_most_recent_ckpt(model=model, optimizer=optimizer)
    else:
        shutil.rmtree(artifacts_dir, ignore_errors=True)
        os.makedirs(artifacts_dir)

    model.compile(
        run_eagerly=True,
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(name='mse'),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.CosineSimilarity(name='cos')
        ])

    callbacks = [
        terminate_on_nan_callback,
        checkpoint_callback,
        ram_cleaner_callback]

    model.fit(
        ds_train,
        epochs=nr_of_epochs,
        verbose=2,
        validation_data=ds_val,
        initial_epoch=starting_epoch,
        shuffle=False,
        callbacks=callbacks)

    upload_model(model=model, model_id=model_id)

    print('Training completed.')
