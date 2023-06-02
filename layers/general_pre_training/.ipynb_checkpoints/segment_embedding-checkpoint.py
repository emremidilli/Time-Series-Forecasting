# Input contains multiple segments. 
# A segment represents wheather input token represents a lookback patch, a forecast patch or a special token (e.g., [SEP],[CLS] etc.).
# Each segment is embedded seperately.

import tensorflow as tf

import numpy as np

class Segment_Embedding(tf.keras.layers.Layer):
    
    def __init__(self,iEmbeddingDims, iNrOfChannels, iNrOfLookbackPatches, iNrOfForecastPatches,**kwargs):
        
        super().__init__(**kwargs)
        
        self.iEmbeddingDims = iEmbeddingDims
        
        self.iNrOfChannels = iNrOfChannels
        self.iNrOfLookbackPatches = iNrOfLookbackPatches
        self.iNrOfForecastPatches = iNrOfForecastPatches
        self.iNrOfFeaturesPerChannel = (iNrOfLookbackPatches  + iNrOfForecastPatches) + 4
        self.iNrOfPositions = self.iNrOfChannels * self.iNrOfFeaturesPerChannel
        
        self.aSegmentBase = self.aGetSegmentBase()
        
        
        self.oDense = tf.keras.layers.Dense(
            units = self.iEmbeddingDims,
            name = 'dense_segment_embedding'
            )
        
        self.oEmbedding = tf.keras.layers.Embedding(
            input_dim= 3 + 1, # {lookback, special, forcast}
            input_length=self.iNrOfPositions,
            output_dim= self.iEmbeddingDims 
            )
        

        
    def aGetSegmentBase(self):
        # lookback base: 1
        # special token base: 2
        # forecast base: 3
        
        
        aSegmentBase = np.ones(shape = self.iNrOfPositions) + 1 # default special tokens (2)
        
        for i in range(self.iNrOfChannels):
            # cls: beginning of each channel.
            iFirstTokenIndex = i * self.iNrOfFeaturesPerChannel 
            
            # lookback window: after cls 
            iLookbackStartIndex = iFirstTokenIndex+1
            iLookbackEndIndex = iLookbackStartIndex + self.iNrOfLookbackPatches - 1
            
            # forecast window: 
            iForecastStartIndex = iLookbackEndIndex+2 # (there is [SEP] between end of lookback and start of forecast)
            iForecastEndIndex = iForecastStartIndex + self.iNrOfForecastPatches - 1
            
            
            aSegmentBase[iLookbackStartIndex:iLookbackEndIndex+1] = 1 
            aSegmentBase[iForecastStartIndex:iForecastEndIndex+1] = 3
            
            
        return aSegmentBase
    
    
    def call(self, x):
        
        y = self.oEmbedding(self.aSegmentBase)
        y = y + self.oDense(x) 
        
        return y
    

