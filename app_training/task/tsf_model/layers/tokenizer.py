import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class PatchTokenizer(tf.keras.layers.Layer):
    '''
    A patch tokenizer that reshapes sub-series in patches.
    '''
    def __init__(self, patch_size, nr_of_covariates, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.trainable = False
        self.nr_of_covariates = nr_of_covariates

        self.reshaper = tf.keras.layers.Reshape(
            (-1, patch_size * nr_of_covariates))

    def call(self, x):
        '''
        inputs:
            x: (None, timesteps, covariates)
        outputs:
            y: (None, nr_of_patches, patch_size x nr_of_covariates)
                in rest of the model, it will be treated as
                (None, timesteps, features)
        '''
        y = self.reshaper(x)
        return y
