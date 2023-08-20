import tensorflow as tf


class PositionEmbedding(tf.keras.layers.Layer):
    '''
    LSTM layer that processes sequential input.
    '''

    def __init__(self, iUnits, **kwargs):

        super().__init__(**kwargs)

        self.lstm = tf.keras.layers.LSTM(units=iUnits,
                                         return_sequences=True)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        '''
        input: (None, timesteps, feature)
        output: (None, timesteps, iUnits)
        '''

        y = self.lstm(x)

        y = self.layer_norm(y)

        return y


class Time2Vec(tf.keras.layers.Layer):
    '''Embedds a datetime vector via perioid activation based on
        "Time2Vec: Learning a Vector Representation of Time" paper.'''
    def __init__(self, kernel_size, periodic_activation='sin', **kwargs):
        super(Time2Vec, self).__init__(
            trainable=True,
            **kwargs
        )

        self.k = kernel_size - 1
        self.p_activation = periodic_activation

    def build(self, input_shape):
        self.wb = self.add_weight(
            shape=(1, 1),
            initializer='uniform',
            trainable=True
        )

        self.bb = self.add_weight(
            shape=(1, 1),
            initializer='uniform',
            trainable=True
        )

        self.wa = self.add_weight(
            shape=(1, self.k),
            initializer='uniform',
            trainable=True
        )

        self.ba = self.add_weight(
            shape=(1, self.k),
            initializer='uniform',
            trainable=True
        )

        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        '''
        inputs: (None, feature_size, 1)
        returns: (None, feature_size, kernel_size)
        '''
        bias = self.wb * inputs + self.bb
        if self.p_activation.startswith('sin'):
            wgts = tf.keras.backend.sin(
                tf.keras.backend.dot(inputs, self.wa) + self.ba)
        elif self.p_activation.startswith('cos'):
            wgts = tf.keras.backend.cos(
                tf.keras.backend.dot(inputs, self.wa) + self.ba)
        else:
            raise NotImplementedError('Neither sine or cosine periodic \
                                      activation be selected.')
        return tf.keras.backend.concatenate([bias, wgts], -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.k + 1)
