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

        aMin = tf.math.reduce_min(x_lb, axis=1)
        aMax = tf.math.reduce_max(x_lb, axis=1)
        y = tf.subtract(aMax, aMin)
        z = tf.subtract(x, tf.expand_dims(aMin, axis=1))
        r = tf.divide(z, tf.expand_dims(y, axis=1))

        return r


class LayerNormalizer(tf.keras.layers.Layer):
    '''
    Used to apply layer normalization to distribution, trend and seasonility
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layer_normalizer_dist = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layer_normalizer_tre = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layer_normalizer_sea = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)

    def call(self, inputs, training=True):
        '''
            inputs: tuple of 3 elements.
                1. distribution (None, timesteps, features)
                2. trend (None, timesteps, features)
                3. seasonility (None, timesteps, features)

            outputs: tuple of the normalized elements
                1. distribution (None, timesteps, features)
                2. trend (None, timesteps, features)
                3. seasonility (None, timesteps, features)
        '''
        dist, tre, sea = inputs

        dist = self.layer_normalizer_dist(dist)
        tre = self.layer_normalizer_tre(tre)
        sea = self.layer_normalizer_sea(sea)

        return (dist, tre, sea)


class BatchNormalizer(tf.keras.layers.Layer):
    '''Batch normalization for date features'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-6)

    def call(self, inputs, training=True):
        '''
        inputs: (none, features)
        outputs: (none, features)
        '''
        y = self.batch_norm(inputs)

        return y