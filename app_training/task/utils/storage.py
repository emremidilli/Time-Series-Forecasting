'''used to integrate with AWS S3 bucket'''

import boto3

import mlflow

import os

from pathlib import Path

import shutil

import tarfile

import tensorflow as tf


def log_experiments(model_id, history, parameters):
    '''
    Experiments are logged into Databricks with MlFlow.
    args:
        model_id (str)
        history (pandas.DataFrame)
        parameters (dictionary)
    '''

    mlflow.login()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(f'/{model_id}')

    with mlflow.start_run():
        history_logs = history.history
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

    shutil.rmtree(os.path.join('tmp'), ignore_errors=True)


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

    Path(os.path.join('tmp'))\
        .mkdir(parents=True, exist_ok=True)

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

    shutil.rmtree(os.path.join('tmp'), ignore_errors=True)

    return model
