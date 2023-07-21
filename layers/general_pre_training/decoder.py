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
    


class ProjectionHead(tf.keras.layers.Layer):
    
    def __init__(self,iFfnUnits ,**kwargs):
        super().__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten()
        
        self.dense = tf.keras.layers.Dense(units=iFfnUnits, activation='tanh')



    
    def call(self, x):
        '''
            decodes the input to the temporal sequence.

            input: (None, timesteps, feature)
            output: (None, feature)
        '''
        
        x = self.flatten(x)
        
        y = self.dense(x)
        
        return y
    

class DistanceLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

