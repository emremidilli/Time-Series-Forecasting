'''
    In original MultiHeadAttention layer of Keras, for each head a different values are used.
    This makes interpretability difficult.
    That's why the authors of TFT proposed InterpretableMultiHeadAttention where values are shared accross heads.
'''

import tensorflow as tf

class interpretable_multi_head_attention(tf.keras.layers.Layer):
    
    def __init__(self,iNrOfHeads, iModelDims, fDropout, **kwargs):
        super().__init__(**kwargs)
        
        self.iNrOfHeads = iNrOfHeads
        self.iModelDims = iModelDims
        
        self.iKeyDims = self.iModelDims // self.iNrOfHeads
        self.iValueDims = self.iModelDims // self.iNrOfHeads
        
        
        self.aQueries = []
        self.aKeys = []
        self.aValues = []
        oSharedValue = tf.keras.layers.Dense(units = self.iValueDims, use_bias = False, activation = None)
        for i in range(self.iNrOfHeads): 
            self.aQueries.append(
                tf.keras.layers.Dense(units = self.iKeyDims, use_bias = False, activation = None)
            )
            
            self.aKeys.append(
                tf.keras.layers.Dense(units = self.iKeyDims, use_bias = False, activation = None)
            )
            
            self.aValues.append(oSharedValue)
            
        
        self.oAttention = tf.keras.layers.Attention(use_scale = True)
        self.oDropout = tf.keras.layers.Dropout(rate = fDropout)
        self.oDense = tf.keras.kayers.Dense(units = iKeyDims, use_bias = False)
        

    def call(self, q, k, v, mask = None):
        
        
        aHeads = []
        aAttentions = []
        
        for i in range(self.iNrOfHeads):
            q_i = self.aQueries[i](q)
            k_i = self.aKeys[i](k)
            v_i = self.aValues[i](v)
            
            
            head , attn = self.oAttention(q_i, k_i, v_i, mask)
            head_dropped = self.oDropout(head)
            
            aHeads.append(head)
            aAttentions.append(attn)
            
        
        if self.iNrOfHeads > 1:
            head = tf.stack(aHeads)
        else:
            head = aHeads[0]
        
        attn = tf.stack(aAttentions)
        
        if self.iNrOfHeads > 1:
            y = tf.mean(head, axis = 0)
        else:
            y = head
            
            
        y = self.oDense(y)
        y = self.oDropout(y)
        
        
        return y, attn
    