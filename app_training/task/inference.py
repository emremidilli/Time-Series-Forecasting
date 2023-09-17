import tensorflow as tf

from utils import get_inference_args

if __name__ == '__main__':
    '''Produces prediction based on given dataset.'''
    args = get_inference_args()

    input_dataset_dir = args.input_dataset_dir
    model_dir = args.model_dir
    output_save_dir = args.output_save_dir

    ds = tf.data.Dataset.load(path=input_dataset_dir)
    predictor = tf.keras.models.load_model(model_dir)

    ds = ds.batch(len(ds))

    numpy_inputs = list(ds.as_numpy_iterator())[0]

    pred = predictor.predict(numpy_inputs)

    ds_pred = tf.data.Dataset.from_tensor_slices(pred)

    ds_pred.save(output_save_dir)
