import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class PatchMasker(tf.keras.layers.Layer):

    def __init__(self, masking_rate, msk_scalar, **kwargs):
        super().__init__(**kwargs)

        self.masking_rate = masking_rate
        self.msk_scalar = msk_scalar
        self.trainable = False

    def call(self, inputs):
        '''
        inputs: tuple of 3 elements
            1. x_tre: (None, timesteps, feature)
            2. x_sea: (None, timesteps, feature)
            3. x_res: (None, timesteps, feature)

        maskes some patches randomly. Each aspect is masked in the same

        outputs: tuple of 3 masked elements
        '''

        x_tre, x_sea, x_res = inputs

        nr_of_timesteps = x_tre.shape[1]
        nr_of_timesteps_to_mask = int(tf.math.ceil(
            nr_of_timesteps * self.masking_rate))

        random_tensor = \
            tf.random.uniform(shape=(nr_of_timesteps, ), minval=0, maxval=1)

        sorted_indices = tf.argsort(random_tensor)

        indices_to_mask = sorted_indices[: nr_of_timesteps_to_mask]

        y_tre = tf.add(tf.zeros_like(x_tre), self.msk_scalar)
        y_sea = tf.add(tf.zeros_like(x_sea), self.msk_scalar)
        y_res = tf.add(tf.zeros_like(x_res), self.msk_scalar)

        z_tre = []
        z_sea = []
        z_res = []
        z_masks = []
        for i in range(nr_of_timesteps):

            r_tre = tf.constant([])
            r_sea = tf.constant([])
            r_res = tf.constant([])
            r_mask = tf.constant(False)

            if tf.reduce_any(tf.math.equal(i, indices_to_mask)):
                r_tre = y_tre[:, i]
                r_sea = y_sea[:, i]
                r_res = y_res[:, i]
                r_mask = True
            else:
                r_tre = x_tre[:, i]
                r_sea = x_sea[:, i]
                r_res = x_res[:, i]
                r_mask = False

            z_tre.append(r_tre)
            z_sea.append(r_sea)
            z_res.append(r_res)
            z_masks.append(r_mask)

        z_tre = tf.stack(z_tre, axis=1)
        z_sea = tf.stack(z_sea, axis=1)
        z_res = tf.stack(z_res, axis=1)
        z_masks = tf.stack(z_masks)

        return (z_tre, z_sea, z_res, z_masks)
