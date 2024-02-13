import tensorflow as tf

from utils import get_inference_args, load_model

if __name__ == '__main__':
    '''Produces prediction based on given dataset.'''
    args = get_inference_args()

    model_id = args.model_id
    input_dir = args.input_dir
    output_dir = args.output_dir

    ds = tf.data.Dataset.load(path=input_dir)

    predictor = load_model(model_id=model_id)

    ds = ds.batch(len(ds))

    ds_input = list(ds.as_numpy_iterator())[0]

    pred = predictor.predict(ds_input)

    ds_pred = tf.data.Dataset.from_tensor_slices(pred)

    ds_pred.save(output_dir)
