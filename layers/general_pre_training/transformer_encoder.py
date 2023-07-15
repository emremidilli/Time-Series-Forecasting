import tensorflow as tf


class TransformerEncoder(tf.keras.layers.Layer):
    
    def __init__(self, iKeyDims, iNrOfHeads, fDropoutRate, iFfnUnits, iFeatureSize , **kwargs):
        super().__init__(**kwargs)
        
        
        # 1st part of the encoder is multi-head attention mechanism
        self.oLayerNorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.oMha_1 = tf.keras.layers.MultiHeadAttention(key_dim=iKeyDims, num_heads=iNrOfHeads, dropout=fDropoutRate)        
        self.oDropOut_1 = tf.keras.layers.Dropout(fDropoutRate)
        
        # 2nd part of the encoder is fully connected networ.
        self.oLayerNorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.oDense_2 = tf.keras.layers.Dense(units=iFfnUnits, activation='relu')
        self.oDropOut_2 = tf.keras.layers.Dropout(fDropoutRate)
        
        # 3rd oart of the encoder is to connect an encoder block to the next one.
        self.oDense_3 = tf.keras.layers.Dense(units=iFeatureSize)
    



    def call(self, x):
        
        x_input = x
        
        # 1st part
        x = self.oLayerNorm_1(x)
        x = self.oMha_1(x, x) #self-attention
        x = self.oDropOut_1(x)
        
        residual = x_input  + x #residual connection
        
        # 2nd part
        x = self.oLayerNorm_1(residual)
        x = self.oDense_2(x)
        x = self.oDropOut_2(x)
        
        # 3rd part
        x = self.oDense_3(x)
        
        return x + residual        

        
        
        
        
        
        
        