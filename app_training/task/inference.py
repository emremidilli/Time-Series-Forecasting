import tensorflow as tf

from utils import get_inference_args, load_model

if __name__ == '__main__':
    '''Produces prediction based on given dataset.'''
    args = get_inference_args()

    dataset_id = args.dataset_id
    model_id = args.model_id

    dataset_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['PREPROCESSED_NAME'],
        dataset_id,
        'dataset')

    ds = tf.data.Dataset.load(path=dataset_dir)

    predictor = load_model(model_id=model_id)

    ds = ds.batch(len(ds))

    ds_input = list(ds.as_numpy_iterator())[0]

    pred = predictor.predict(ds_input)

    ds_pred = tf.data.Dataset.from_tensor_slices(pred)

    output_save_dir = os.path.join(
        os.environ['BIN_NAME'],
        os.environ['INFERENCE_NAME'],
        dataset_id)

    ds_pred.save(output_save_dir)
