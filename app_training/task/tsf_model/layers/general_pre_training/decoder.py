import tensorflow as tf


class MppDecoder(tf.keras.layers.Layer):
    '''
    Decoder for masked patch prediction task.
    '''

    def __init__(self, iFfnUnits, iNrOfTimeSteps, **kwargs):
        super().__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(
            units=iFfnUnits * iNrOfTimeSteps,
            use_bias=False)

        self.reshape = tf.keras.layers.Reshape(
            target_shape=(iNrOfTimeSteps, iFfnUnits))

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

    def call(self, x):
        '''
        input: (None, timesteps, feature)
        output: (None, feature)
        '''

        x = self.flatten(x)

        y = self.dense(x)

        return y
