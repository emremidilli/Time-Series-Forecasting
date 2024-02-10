import tensorflow as tf

from utils import read_npy_file, get_input_args_inference


if __name__ == '__main__':
    '''
    Receives directories for numpy inputs of
    1. lookback
    2. date_features
    Pre-processes them by a pre-processor based on given model_id.
    Pre-processed data is saved in tf.data.Dataset format.
    '''

    args = get_input_args_inference()
    print(args)

    lb_dir = args.lb_dir
    ts_dir = args.ts_dir
    pre_processor_dir = args.pre_processor_dir
    save_dir = args.save_dir

    lb = read_npy_file(lb_dir, dtype='float32')
    ts = read_npy_file(ts_dir, dtype='int32')

    input_pre_processor = tf.saved_model.load(pre_processor_dir)

    tre, sea, res, ts = input_pre_processor((lb, ts))

    ds = tf.data.Dataset.from_tensor_slices((tre, sea, res, ts))

    ds.save(save_dir)
