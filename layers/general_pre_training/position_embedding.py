# Embedds according to position encoding of Attention is All You Need paper.
# In original paper, input tokens are words. That's why, it contains embedding layer.
# In this study, input tokens are vectors which we can consider already an embedded outcomes.
# We feed to dense layer to project in accordance with model dims.
# Output of dense is summed with the sinusoidal base.

import tensorflow as tf

import numpy as np

class Position_Embedding(tf.keras.layers.Layer):
    
    def __init__(self, iNrOfPositions, iEmbeddingDims, **kwargs):
        
        super().__init__(**kwargs)
        
        self.iNrOfPositions = iNrOfPositions
        self.iEmbeddingDims = iEmbeddingDims
        
        
        self.aSinusoidalBase = self.aGetSinusoidalBase(
            iLength=self.iNrOfPositions, 
            iDepth=self.iEmbeddingDims
            )
        
        
        self.oDense = tf.keras.layers.Dense(
            units = self.iEmbeddingDims,
            name = 'dense_position_embedding'
            )
        
        

    def aGetSinusoidalBase(self,  iLength, iDepth):

        iDepthToReturn = iDepth

        iDepth = iDepth/2

        aPositions = np.arange(iLength)[:, np.newaxis]     # (seq, 1)

        aDepths = np.arange(iDepth)[np.newaxis, :]/iDepth   # (1, iDepth)

        aAngleRates = 1 / (10000**aDepths)         # (1, iDepth)
        angle_rads = aPositions * aAngleRates      # (pos, iDepth)

        aSinusoidalBase = np.concatenate(
          [np.sin(angle_rads), np.cos(angle_rads)],
          axis=-1) 

        aSinusoidalBase = aSinusoidalBase[:, :iDepthToReturn]

        return tf.cast(aSinusoidalBase, dtype=tf.float32)
    
    
    
    def call(self, x):
        
        y = self.oDense(x)
        y = y + self.aSinusoidalBase
        
        return y
    
    
    

    
    