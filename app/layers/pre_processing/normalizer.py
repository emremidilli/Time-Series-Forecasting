import tensorflow as tf

class LookbackNormalizer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False


    def call(self, inputs):
        '''
            inputs: tuple of 2 elements.
                1. original lookback series (None, nr_of_lookback_time_steps)
                2. series to normalize (None, nr_of_time_steps)

            outputs: the normalized series (None, nr_of_time_steps)
        '''

        x_lb, x = inputs

        aMin = tf.math.reduce_min(x_lb, axis =1)
        aMax = tf.math.reduce_max(x_lb, axis =1)
        y = tf.subtract(aMax, aMin)
        z = tf.subtract(x, tf.expand_dims(aMin, axis = 1))
        r = tf.divide(z, tf.expand_dims(y, axis = 1))

        return r
