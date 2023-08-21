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
        "Time2Vec: Learning a Vector Representation of Time" paper.
        As difference from original paper:
            In order to prevent from gradient explosion for higher dimensions,
            layer normalization is employed at end of the layer.
    '''
    def __init__(self, embedding_dims, **kwargs):
        super(Time2Vec, self).__init__(**kwargs)

        self.dense_linear = tf.keras.layers.Dense(
            units=1,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform'
        )

        self.dense_periodic = tf.keras.layers.Dense(
            units=embedding_dims - 1,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform'
        )

        self.concatter = tf.keras.layers.Concatenate(axis=2)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, **kwargs):
        '''
        inputs: (None, feature_size)
        returns: (None, feature_size, embedding_dims)
        '''
        x = tf.expand_dims(inputs, axis=2)

        linear = self.dense_linear(x)

        periodic = self.dense_periodic(x)
        periodic = tf.keras.backend.sin(
            periodic
        )

        embedded = self.concatter([linear, periodic])

        y = self.layer_norm(embedded)

        return y
