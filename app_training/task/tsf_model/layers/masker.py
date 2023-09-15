import tensorflow as tf


class PatchMasker(tf.keras.layers.Layer):

    def __init__(self, masking_rate, msk_scalar, **kwargs):
        super().__init__(**kwargs)

        self.masking_rate = masking_rate
        self.msk_scalar = msk_scalar
        self.trainable = False

    def call(self, inputs):
        '''
        inputs: tuple of 3 elements
            1. x_dist: (None, timesteps, feature)
            2. x_tre: (None, timesteps, feature)
            3. x_sea: (None, timesteps, feature)

        maskes some patches randomly. Each aspect is masked in the same

        outputs: tuple of 3 masked elements
        '''

        x_dist, x_tre, x_sea = inputs

        iNrOfPatches = x_dist.shape[1]

        y_dist = tf.add(tf.zeros_like(x_dist), self.msk_scalar)
        y_tre = tf.add(tf.zeros_like(x_tre), self.msk_scalar)
        y_sea = tf.add(tf.zeros_like(x_sea), self.msk_scalar)

        z_dist = []
        z_tre = []
        z_sea = []
        for i in range(iNrOfPatches):

            r_dist = tf.constant([])
            r_tre = tf.constant([])
            r_sea = tf.constant([])

            fRand = tf.random.uniform(
                shape=[], minval=0, maxval=1, dtype=tf.float32)
            if fRand <= self.masking_rate:
                r_dist = y_dist[:, i]
                r_tre = y_tre[:, i]
                r_sea = y_sea[:, i]
            else:
                r_dist = x_dist[:, i]
                r_tre = x_tre[:, i]
                r_sea = x_sea[:, i]

            z_dist.append(r_dist)
            z_tre.append(r_tre)
            z_sea.append(r_sea)

        z_dist = tf.stack(z_dist, axis=1)
        z_tre = tf.stack(z_tre, axis=1)
        z_sea = tf.stack(z_sea, axis=1)

        return (z_dist, z_tre, z_sea)
