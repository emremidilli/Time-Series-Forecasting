import tensorflow as tf


class Rpp_Decoder(tf.keras.layers.Layer):
    
    def __init__(self,iFfnUnits, **kwargs):
        super().__init__(**kwargs)
        
        self.oDense = tf.keras.layers.Dense(units=iFfnUnits, activation='softmax') # since output of RPP are one-hot encoded.
    
    def call(self, x):
        x = self.oDense (x)
        
        return x