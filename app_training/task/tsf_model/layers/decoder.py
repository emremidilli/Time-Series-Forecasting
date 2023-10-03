import tensorflow as tf


class MppDecoder(tf.keras.layers.Layer):
    '''Decoder for masked patch prediction task.'''
    def __init__(self, iFfnUnits, nr_of_time_steps, **kwargs):
        super().__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(
            units=iFfnUnits * nr_of_time_steps,
            use_bias=False)

        self.layer_norm = tf.keras.layers.LayerNormalization()

        self.reshape = tf.keras.layers.Reshape(
            target_shape=(nr_of_time_steps, iFfnUnits))

    def call(self, x):
        '''
        input: (None, timesteps, feature)
        output: (None, timesteps, feature)
        '''

        x = self.flatten(x)

        y = self.dense(x)
        y = self.layer_norm(y)
        y = self.reshape(y)

        return y


class ProjectionHead(tf.keras.layers.Layer):
    '''Projection head for contrastive learning task.'''
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


class SingleStepDecoder(tf.keras.layers.Layer):
    '''Decoder for singe-step predictor.'''
    def __init__(
            self,
            nr_of_time_steps,
            alpha_regulizer=0.20,
            l1_ratio=0.50,
            **kwargs):
        super().__init__(**kwargs)

        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=nr_of_time_steps * 3),
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(units=nr_of_time_steps * 2),
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(units=nr_of_time_steps),
            tf.keras.layers.Reshape(target_shape=(nr_of_time_steps, 1))
        ])

        # self.dense_1 = tf.keras.layers.Dense(
        #     units=nr_of_time_steps * 3,
        #     kernel_regularizer=tf.keras.regularizers.L1L2(
        #         l1=l1_ratio * alpha_regulizer,
        #         l2=(1 - l1_ratio) * alpha_regulizer))

        # self.dense_2 = tf.keras.layers.Dense(
        #     units=nr_of_time_steps * 2,
        #     kernel_regularizer=tf.keras.regularizers.L1L2(
        #         l1=l1_ratio * alpha_regulizer,
        #         l2=(1 - l1_ratio) * alpha_regulizer))

        # self.dense_3 = tf.keras.layers.Dense(
        #     units=nr_of_time_steps,
        #     kernel_regularizer=tf.keras.regularizers.L1L2(
        #         l1=l1_ratio * alpha_regulizer,
        #         l2=(1 - l1_ratio) * alpha_regulizer))

    def call(self, x):
        '''
        input: (None, timesteps, feature)
        output: (None, timesteps, 1)
        '''

        y = self.feed_forward(x)

        return y
