import tensorflow as tf


class gated_linear_unit(tf.keras.layers.Layer):

    def __init__(self, iFfnUnits, **kwargs):
        super().__init__(**kwargs)

        self.oDense_a = tf.keras.layers.Dense(units=iFfnUnits, activation=None)

        self.oDense_b = tf.keras.layers.Dense(units=iFfnUnits,
                                              activation='sigmoid')
        self.oMultiplier = tf.keras.layers.Multiply()

    def call(self, x):

        b = self.oDense_b(x)

        a = self.oDense_a(x)

        y = self.oMultiplier([a, b])

        return y
