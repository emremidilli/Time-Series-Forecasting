

import tensorflow as tf

import numpy as np


class Channel_Embedding(tf.keras.layers.Layer):
    
    def __init__(self, iEmbeddingDims, iNrOfChannels, iNrOfLookbackPatches, iNrOfForecastPatches, **kwargs):
        
        super().__init__(**kwargs)
        
        self.iEmbeddingDims = iEmbeddingDims
        
        self.iNrOfChannels = iNrOfChannels
        self.iNrOfLookbackPatches = iNrOfLookbackPatches
        self.iNrOfForecastPatches = iNrOfForecastPatches
        self.iNrOfFeaturesPerChannel = (iNrOfLookbackPatches  + iNrOfForecastPatches) + 4
        self.iNrOfPositions = self.iNrOfChannels * self.iNrOfFeaturesPerChannel
        
        
        self.aChannelBase= self.aGetChannelBase()
        
        self.oDense = tf.keras.layers.Dense(
            units = self.iEmbeddingDims,
            name = 'dense_channel_embedding'
            )
        
        self.oEmbedding = tf.keras.layers.Embedding(
            input_dim= self.iNrOfChannels + 1,
            input_length=self.iNrOfPositions,
            output_dim= self.iEmbeddingDims 
            )
        
        
        
    def aGetChannelBase(self):
        
        aChannelBase= np.zeros(shape = self.iNrOfPositions)
        
        for i in range(self.iNrOfChannels):
            # cls: beginning of each channel.
            iFirstTokenIndex = i * self.iNrOfFeaturesPerChannel 
                        
            aChannelBase[iFirstTokenIndex:] = i + 1 
                
        return aChannelBase
        
    
    
    def call(self, x):
        
        y = self.oEmbedding(self.aChannelBase)
        y = y + self.oDense(x) 
        
        return y