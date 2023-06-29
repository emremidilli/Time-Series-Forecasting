import tensorflow as tf


class Npp_Decoder(tf.keras.layers.Layer):
    
    def __init__(self,iFfnUnits, **kwargs):
        super().__init__(**kwargs)
        
        self.oFlatten = tf.keras.layers.Flatten()
        self.oDense = tf.keras.layers.Dense(units=iFfnUnits, activation='sigmoid')
        
        
    def call(self, x):
        
        x = self.oFlatten(x)
        x = self.oDense (x)
        
        return x
