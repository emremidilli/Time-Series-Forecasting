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

    ds_input, ds_stat = list(ds.as_numpy_iterator())[0]

    pred = predictor.predict(ds_input)

    min_lb, max_lb = ds_stat
    min_lb = tf.expand_dims(tf.expand_dims(min_lb, 1), 1)
    max_lb = tf.expand_dims(tf.expand_dims(max_lb, 1), 1)

    pred_inverse_normalized = ((max_lb - min_lb) * pred) + min_lb

    ds_pred = tf.data.Dataset.from_tensor_slices(pred_inverse_normalized)

    ds_pred.save(output_save_dir)
