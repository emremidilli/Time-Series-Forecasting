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
        nr_of_timesteps_to_mask = tf.cast(
            tf.math.ceil(nr_of_timesteps * self.masking_rate),
            dtype=tf.int32)

        random_tensor = \
            tf.random.uniform(shape=(nr_of_timesteps, ), minval=0, maxval=1)

        sorted_indices = tf.argsort(random_tensor)

        indices_to_mask = sorted_indices[: nr_of_timesteps_to_mask]

        y_tre = tf.add(tf.zeros_like(x_tre), self.msk_scalar)
        y_sea = tf.add(tf.zeros_like(x_sea), self.msk_scalar)
        y_res = tf.add(tf.zeros_like(x_res), self.msk_scalar)

        mask_condition = tf.reduce_any(
            tf.equal(
                tf.range(nr_of_timesteps)[:, tf.newaxis],
                indices_to_mask), axis=1)
        # Expand dimensions for broadcasting
        mask_condition = tf.expand_dims(mask_condition, axis=-1)

        r_tre = tf.where(mask_condition, y_tre, x_tre)
        r_sea = tf.where(mask_condition, y_sea, x_sea)
        r_res = tf.where(mask_condition, y_res, x_res)

        z_tre = tf.stack([r_tre[:, i] for i in range(nr_of_timesteps)], axis=1)
        z_sea = tf.stack([r_sea[:, i] for i in range(nr_of_timesteps)], axis=1)
        z_res = tf.stack([r_res[:, i] for i in range(nr_of_timesteps)], axis=1)
        z_masks = tf.squeeze(mask_condition, axis=-1)

        return (z_tre, z_sea, z_res, z_masks)
