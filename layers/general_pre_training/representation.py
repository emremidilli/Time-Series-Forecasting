import tensorflow as tf

import sys
sys.path.append( '../')

from layers.general_pre_training.channel_embedding import Channel_Embedding

from layers.general_pre_training.segment_embedding import Segment_Embedding

from layers.general_pre_training.position_embedding import Position_Embedding

from layers.general_pre_training.transformer_encoder import Transformer_Encoder


class Representation(tf.keras.layers.Layer):
    def __init__(self, iNrOfChannels , iNrOfQuantiles,iNrOfLookbackPatches, iNrOfForecastPatches  ,iNrOfEncoderBlocks,iNrOfHeads,fDropoutRate, iEncoderFfnUnits,iEmbeddingDims,  **kwargs):
        super().__init__(**kwargs)
        
        self.iNrOfEncoderBlocks = iNrOfEncoderBlocks
        self.iNrOfHeads = iNrOfHeads
        self.fDropoutRate = fDropoutRate
        self.iEncoderFfnUnits = iEncoderFfnUnits
        self.iEmbeddingDims = iEmbeddingDims
        
        
        self.iNrOfChannels = iNrOfChannels
        self.iNrOfQuantiles = iNrOfQuantiles

        iNrOfLookbackPatches = iNrOfLookbackPatches
        iNrOfForecastPatches = iNrOfForecastPatches
        iNrOfPositions = self.iNrOfChannels * (iNrOfLookbackPatches  + iNrOfForecastPatches)
        
        
        self.ce = Channel_Embedding(
            iEmbeddingDims = iEmbeddingDims,
            iNrOfChannels = self.iNrOfChannels, 
            iNrOfLookbackPatches = iNrOfLookbackPatches, 
            iNrOfForecastPatches = iNrOfForecastPatches
            )
        
        self.pe = Position_Embedding(iNrOfPositions, iEmbeddingDims)
        
        self.se = Segment_Embedding(iEmbeddingDims, self.iNrOfChannels, iNrOfLookbackPatches, iNrOfForecastPatches)
        
        self.aTransformerEncoders = []
        for i in range(iNrOfEncoderBlocks):
            oToAdd = Transformer_Encoder(
                iNrOfPositions = iNrOfPositions, 
                iKeyDims = iEmbeddingDims, 
                iNrOfHeads = iNrOfHeads, 
                fDropoutRate = fDropoutRate, 
                iFfnUnits = iEncoderFfnUnits
            )
            self.aTransformerEncoders.append(oToAdd)    
                
    def call(self, x):
        
        x = self.ce(x) + self.pe(x) + self.se(x)

        for oEncoder in self.aTransformerEncoders:
            x = oEncoder(x)

        return x                

    

    