import tensorflow as tf

from utils import get_inference_args

if __name__ == '__main__':
    '''Produces prediction based on given dataset.'''
    args = get_inference_args()

    input_dataset_dir = args.input_dataset_dir
    model_dir = args.model_dir
    output_save_dir = args.output_save_dir
    nr_of_forecasting_steps = args.nr_of_forecasting_steps
    begin_scalar = args.begin_scalar

    ds = tf.data.Dataset.load(path=input_dataset_dir)
    predictor = tf.keras.models.load_model(model_dir)

    ds = ds.batch(len(ds))

    ds_input, ds_stat = list(ds.as_numpy_iterator())[0]

    t = predictor.con_temp_pret(ds_input)

    begin = tf.zeros(
        (t.shape[0], nr_of_forecasting_steps + 1, 1)) + begin_scalar

    input = begin
    pred = None
    for i in range(nr_of_forecasting_steps):
        z = predictor.pe(input)
        y = predictor.decoder((z, t))
        y = predictor.dense(y)

        y = y[:, i, :]
        y = tf.expand_dims(y, axis=1)

        if i == 0:
            pred = y
        else:
            pred = tf.concat([pred, y], axis=1)

    min_lb, max_lb = ds_stat
    min_lb = tf.expand_dims(tf.expand_dims(min_lb, 1), 1)
    max_lb = tf.expand_dims(tf.expand_dims(max_lb, 1), 1)

    pred_inverse_normalized = ((max_lb - min_lb) * pred) + min_lb

    ds_pred = tf.data.Dataset.from_tensor_slices(pred_inverse_normalized)

    ds_pred.save(output_save_dir)
