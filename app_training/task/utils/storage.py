'''used to integrate with AWS S3 bucket'''

import boto3

import mlflow

import os

from pathlib import Path

import shutil

import tarfile

import tensorflow as tf


def _create_tmp_directory():
    '''creates a tmp directory'''
    Path(os.path.join('tmp'))\
        .mkdir(parents=True, exist_ok=True)


def _remove_tmp_directory():
    '''removes tmp directory'''
    shutil.rmtree(os.path.join('tmp'), ignore_errors=True)


def _unbatch_dataset(
        ds: tf.data.Dataset):
    '''covert tf.data.Dataset to single batch tensors'''

    # Reverting back to separate tensors
    tre_list, sea_list, res_list, ts_list = [], [], [], []

    for tre_batch, sea_batch, res_batch, ts_batch in ds:
        tre_list.append(tre_batch)
        sea_list.append(sea_batch)
        res_list.append(res_batch)
        ts_list.append(ts_batch)

    # Concatenate the lists to form tensors
    tre = tf.concat(tre_list, axis=0)
    sea = tf.concat(sea_list, axis=0)
    res = tf.concat(res_list, axis=0)
    ts = tf.concat(ts_list, axis=0)

    new_ds = tf.data.Dataset.from_tensor_slices(
        (tre, sea, res, ts))

    return new_ds


def _predict_and_save(
        model: tf.keras.Model,
        ds: tf.data.Dataset,
        ds_name: str):
    '''
    predicts an input dataset and saves
    the input and prediction into tmp folder.
    '''
    ds = _unbatch_dataset(ds)

    npy_input = list(ds.batch(len(ds)).as_numpy_iterator())[0]

    pred_tre, pred_sea, pred_res, _ = \
        model.predict(npy_input)

    pred_masks = model.masks

    _create_tmp_directory()

    ds_dir = os.path.join('tmp', ds_name)

    true_save_dir = os.path.join(ds_dir, 'true')
    ds.save(true_save_dir)

    pred_save_dir = os.path.join(ds_dir, 'pred')
    pred = tf.data.Dataset.from_tensor_slices((pred_tre, pred_sea, pred_res))
    pred.save(pred_save_dir)

    mask_save_dir = os.path.join(ds_dir, 'masks')
    mask = tf.data.Dataset.from_tensor_slices(pred_masks)
    mask.save(mask_save_dir)

    mlflow.log_artifacts(ds_dir, artifact_path=ds_name)

    _remove_tmp_directory()


def log_experiments(
        model_id: str,
        history: tf.keras.callbacks.History,
        model: tf.keras.Model,
        ds_train: tf.data.Dataset,
        ds_val: tf.data.Dataset,
        ds_test: tf.data.Dataset,
        parameters: dict):
    '''
    Experiments are logged into Databricks with MlFlow.
    '''

    mlflow.login()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(f'/{model_id}')

    with mlflow.start_run():
        history_logs = history.history

        mlflow.log_param("train_evaluation", model.evaluate(ds_train))
        _predict_and_save(model, ds_train, 'train')

        if ds_val is not None:
            mlflow.log_param("validation_evaluation", model.evaluate(ds_val))
            _predict_and_save(model, ds_val, 'validation')

        mlflow.log_param("test_evaluation", model.evaluate(ds_test))
        _predict_and_save(model, ds_test, 'test')

        mlflow.log_param(
            'trainable_params',
            tf.reduce_sum([
                tf.reduce_prod(var.shape)
                for var in model.trainable_variables
            ]))

        mlflow.log_param(
            'non_trainable_params',
            tf.reduce_sum([
                tf.reduce_prod(var.shape)
                for var in model.non_trainable_variables
            ]))

        mlflow.log_params(parameters)
        mlflow.log_table(
            data=history_logs,
            artifact_file="history_logs.json")

        for metric in list(history_logs.keys()):
            mlflow.log_metric(metric, history_logs[metric][-1])


def upload_model(model, model_id):
    '''
    saves model to temp folder on local.
    creates tar.gz of the temp folder.
    uploads the tar.gz model to s3 bucket.
    removes the temp folder.

    args:
        model (tf.keras.Model)
        model_id (str)
    '''
    model_key = os.path.join('model_artifacts', model_id, 'saved_model')

    temp_folderdir = os.path.join('tmp', 'saved_model')
    model.save(
        temp_folderdir,
        overwrite=True,
        save_format='tf')

    output_tar_gz = os.path.join('tmp', 'saved_model.tar.gz')
    with tarfile.open(output_tar_gz, 'w:gz') as tar:
        tar.add(temp_folderdir, arcname='tf_model')

    bucket_name = os.getenv('AWS_STORAGE_BUCKET_NAME')

    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_S3_REGION_NAME'))

    s3.upload_file(
        output_tar_gz,
        bucket_name,
        model_key)

    _remove_tmp_directory()


def load_model(model_id):
    '''
    loads the saved model from s3 bucket in tar.gz format to tmp directory.
    extracts the tar.gz file.
    loads the tensorflow model.
    cleans the tmp directory.

    args:
        model_id (str)

    returns:
        model (tf.keras.Model)
    '''

    _create_tmp_directory()

    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_S3_REGION_NAME'))

    bucket_name = os.getenv('AWS_STORAGE_BUCKET_NAME')
    model_key = os.path.join('model_artifacts', model_id, 'saved_model')
    local_download_path = os.path.join(
        'tmp',
        'dowloaded_model.tar.gz')
    s3.download_file(
        bucket_name,
        model_key,
        local_download_path)

    local_extraction_path = os.path.join(
        'tmp')
    with tarfile.open(local_download_path, 'r:gz') as tar:
        tar.extractall(local_extraction_path)

    model = tf.keras.models.load_model(
        os.path.join(local_extraction_path, 'tf_model'))

    _remove_tmp_directory()

    return model
