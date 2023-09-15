import numpy as np

import tensorflow as tf

from utils import get_args_for_inference

if __name__ == '__main_':
    '''
    receives input and applies pre-processing and produces prediction.
    There are 2 inputs:
        1. lookback: (None, timesteps)
        2. date_features: (None, features)
    Saves predictions in tf.data.Dataset format.
    '''
    args = get_args_for_inference()

    channel = args.channel
    lookback_dir = args.lookback_dir
    date_features_dir = args.date_features_dir
    input_preprocessor_dir = args.input_preprocessor_dir
    predictor_dir = args.predictor_dir
    save_dir = args.save_dir

    lookback = np.load(lookback_dir)
    date_features = np.load(date_features_dir)

    input_pre_processor = tf.keras.models.load(input_preprocessor_dir)

    ds = input_pre_processor((lookback, date_features), training=False)

    predictor = tf.keras.models.load_model(predictor_dir)

    pred = predictor.predict(ds)

    pred.save(save_dir)
