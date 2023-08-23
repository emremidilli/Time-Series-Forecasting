import tensorflow as tf


class PositionEmbedding(tf.keras.layers.Layer):
    '''
    LSTM layer that processes sequential input.
    '''

    def __init__(self, embedding_dims, **kwargs):

        super().__init__(**kwargs)

        self.lstm = tf.keras.layers.LSTM(units=embedding_dims,
                                         return_sequences=True)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        '''
        input: (None, timesteps, features)
        output: (None, timesteps, features)
        '''

        y = self.lstm(x)

        y = self.layer_norm(y)

        return y


# class PositionEmbedding(tf.keras.layers.Layer):
#     '''
#     LSTM layer that processes sequential input.
#     '''

#     class CustomLstmCell(tf.keras.layers.LSTMCell):
#         '''
#         LSTM cell that applies LayerNorm between hidden states.
#         '''
#         def __init__(self, units, **kwargs):
#             self.activation = tf.keras.activations.get(
#                 kwargs.get("activation", "tanh"))
#             kwargs["activation"] = None
#             super().__init__(units, **kwargs)
#             self.layer_norm = tf.keras.layers.LayerNormalization(
#             epsilon=1e-6)

#         def call(self, inputs, states):
#             outputs, new_states = super().call(inputs, states)
#             norm_out = self.activation(self.layer_norm(outputs))
#             return norm_out, [norm_out]

#     def __init__(self, embedding_dims, **kwargs):
#         super().__init__(**kwargs)
#         self.custom_lstm_cell = self.CustomLstmCell(units=embedding_dims)
#         self.lstm = tf.keras.layers.RNN(
#             cell=self.custom_lstm_cell,
#             return_sequences=True)

#         self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

#     def call(self, x):
#         '''
#         input: (None, timesteps, features)
#         output: (None, timesteps, features)
#         '''

#         y = self.lstm(x)

#         y = self.layer_norm(y)

#         return y


# class PositionEmbedding(tf.keras.layers.Layer):
#     def __init__(self, embedding_dims, **kwargs):
#         super().__init__(**kwargs)
#         self.embedding_dims = embedding_dims

#         self.embedding = tf.keras.layers.Dense(
#             units=embedding_dims,
#             kernel_initializer='glorot_uniform',
#             bias_initializer='glorot_uniform')

#         self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

#     def positional_encoding(self, length, depth):
#         import numpy as np

#         depth = depth / 2

#         positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
#         depths = np.arange(depth)[np.newaxis, :] / depth   # (1, depth)

#         angle_rates = 1 / (10000**depths)         # (1, depth)
#         angle_rads = positions * angle_rates      # (pos, depth)

#         pos_encoding = np.concatenate(
#             [np.sin(angle_rads), np.cos(angle_rads)],
#             axis=-1)

#         return tf.cast(pos_encoding, dtype=tf.float32)

#     def call(self, inputs):
#         pos_encodings = self.positional_encoding(
#             length=inputs.shape[1],
#             depth=self.embedding_dims)

#         y = self.embedding(inputs)

#         y = self.layer_norm(y)

#         return y + pos_encodings


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
