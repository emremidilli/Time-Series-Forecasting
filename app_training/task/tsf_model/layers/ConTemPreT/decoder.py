import tensorflow as tf


class MppDecoder(tf.keras.layers.Layer):
    '''
    Decoder for masked patch prediction task.
    '''
    def __init__(self, iFfnUnits, nr_of_time_steps, **kwargs):
        super().__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(
            units=iFfnUnits * nr_of_time_steps,
            use_bias=False)

        self.reshape = tf.keras.layers.Reshape(
            target_shape=(nr_of_time_steps, iFfnUnits))

    def call(self, x):
        '''
        input: (None, timesteps, feature)
        output: (None, timesteps, feature)
        '''

        x = self.flatten(x)

        y = self.dense(x)
        y = self.reshape(y)

        return y


class ProjectionHead(tf.keras.layers.Layer):
    '''
    Projection head for contrastive learning task.
    '''
    def __init__(self, iFfnUnits, **kwargs):
        super().__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(
            units=iFfnUnits,
            activation='relu',
            use_bias=False)

        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        '''
        input: (None, timesteps, feature)
        output: (None, feature)
        '''

        x = self.flatten(x)

        y = self.dense(x)

        y = self.layer_norm(y)

        return y


class QuantileDecoder(tf.keras.layers.Layer):
    '''
    Decoder for quantile predictor.
    '''
    def __init__(self, nr_of_time_steps, nr_of_quantiles, **kwargs):
        super().__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(
            units=nr_of_quantiles * nr_of_time_steps,
            use_bias=False)

        self.reshape = tf.keras.layers.Reshape(
            target_shape=(nr_of_time_steps, nr_of_quantiles))

    def call(self, x):
        '''
        input: (None, timesteps, feature)
        output: (None, timesteps, feature)
        '''

        x = self.flatten(x)

        y = self.dense(x)

        y = self.reshape(y)

        return y
