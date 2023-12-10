import tensorflow as tf


class LookbackNormalizer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def call(self, inputs):
        '''
        inputs: tuple of 2 elements.
            1. x_lb: original lookback series
                (None, nr_of_lookback_time_steps, nr_of_covariates)
            2. x: series to normalize
                (None, nr_of_time_steps, nr_of_covariates)

        returns:
            1. r: the normalized series
                (None, nr_of_time_steps, nr_of_covaraites)
        '''
        x_lb, x = inputs

        aMin = tf.math.reduce_min(x_lb, axis=1)
        aMax = tf.math.reduce_max(x_lb, axis=1)
        y = tf.subtract(aMax, aMin)
        z = tf.subtract(x, tf.expand_dims(aMin, axis=1))
        r = tf.divide(z, tf.expand_dims(y, axis=1))

        return r
