import tensorflow as tf

class PatchShifter(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)




    def call(self, x):
        '''
        inputs: patched input (None, nr_of_patches, feature_size)

        outputs: randomly shifted version (None, nr_of_patches, feature_size)
        '''       
        iNrOfPatches = x.shape[1]

        i = tf.random.uniform(shape=(), minval=1, maxval=iNrOfPatches-1, dtype=tf.int32)

        y = tf.roll(x, shift = i ,axis = 1)

        return y

