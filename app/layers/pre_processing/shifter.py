import tensorflow as tf

class PatchShifter(tf.keras.layers.Layer):
    def __init__(self ,**kwargs):
        super().__init__(**kwargs)
        



    def call(self, inputs):
        '''
        inputs: tuple of 4 elements
            1. x_dist: (None, timesteps, feature)
            2. x_tre: (None, timesteps, feature)
            3. x_sea: (None, timesteps, feature)
            4. shift: integer

        shifts the patch with roll operation.
        
        outputs: tuple of 3 shifted elements
        '''       
        x_dist, x_tre, x_sea, shift = inputs

        y_dist = tf.roll(x_dist, shift = shift ,axis = 1)
        y_tre = tf.roll(x_tre, shift = shift ,axis = 1)
        y_sea = tf.roll(x_sea, shift = shift ,axis = 1)

        return (y_dist, y_tre, y_sea)

