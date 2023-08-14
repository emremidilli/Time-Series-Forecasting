import tensorflow as tf


class PositionEmbedding(tf.keras.layers.Layer):
    '''
    LSTM layer that processes sequential input.
    Output of LSTM layer is normalized accross timesteps.
    '''

    def __init__(self, iUnits, **kwargs):

        super().__init__(**kwargs)

        self.lstm = tf.keras.layers.LSTM(units=iUnits, return_sequences=True)

        self.layer_norm = tf.keras.layers.LayerNormalization(axis=1)

    def call(self, x):
        '''
        input: (None, timesteps, feature)

        output: (None, timesteps, iUnits)
        '''

        y = self.lstm(x)

        y = self.layer_norm(y)

        return y
