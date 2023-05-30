import tensorflow as tf


class Mpp_Decoder(tf.keras.layers.Layer):
    
    def __init__(self,iFfnUnits, **kwargs):
        super().__init__(**kwargs)
        
        self.oDense = tf.keras.layers.Dense(units=iFfnUnits, activation='sigmoid')

    
    def call(self, x):
        
        x = self.oDense (x)
        
        return x
