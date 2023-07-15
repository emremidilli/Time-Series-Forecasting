import tensorflow as tf


class MppDecoder(tf.keras.layers.Layer):
    
    def __init__(self,iFfnUnits, iNrOfTimeSteps ,**kwargs):
        super().__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten()
        
        
        self.dense = tf.keras.layers.Dense(units=iFfnUnits*iNrOfTimeSteps)

        self.reshape = tf.keras.layers.Reshape(target_shape=(iNrOfTimeSteps,iFfnUnits ))


    
    def call(self, x):
        '''
            decodes the input to the temporal sequence.

            input: (None, timesteps, feature)
            output: (None, timesteps, feature)
        '''
        
        x = self.flatten(x)
        
        y = self.dense(x)
        y = self.reshape(y)
        
        
        return y
