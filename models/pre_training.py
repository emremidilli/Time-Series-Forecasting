import tensorflow as tf

import sys
sys.path.append( '../')

from layers.CHANNEL_EMBEDDING import Channel_Embedding

from layers.SEGMENT_EMBEDDING import Segment_Embedding

from layers.POSITION_EMBEDDING import Position_Embedding

from layers.TRANSFORMER_ENCODER import Transformer_Encoder

from layers.NPP_DECODER import Npp_Decoder

from layers.MPP_DECODER import Mpp_Decoder

from layers.SPP_DECODER import Spp_Decoder

from layers.RPP_DECODER import Rpp_Decoder

import preprocessing.constants as c


class Pre_Training(tf.keras.Model):
    
    def __init__(self, sTaskType, **kwargs):
        super().__init__(**kwargs)
        
        self.sTaskType = sTaskType
        self.iNrOfChannels = 3
        self.iNrOfQuantiles = 3
        
        iNrOfEncoderBlocks = 2
        iNrOfHeads = 3
        fDropoutRate = 0.05
        iFfnUnits = 128
        
        iEmbeddingDims = 32

        iNrOfLookbackPatches = 16
        iNrOfForecastPatches = 4
        iNrOfFeaturesPerChannel = (iNrOfLookbackPatches  + iNrOfForecastPatches) + 4
        iNrOfPositions = self.iNrOfChannels * iNrOfFeaturesPerChannel
        
        
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
                iFfnUnits = iFfnUnits)
            self.aTransformerEncoders.append(oToAdd)
        
    
    def build(self, input_shape):
        if self.sTaskType == 'NPP':
            self.oDecoder = Npp_Decoder(
                iFfnUnits = self.iNrOfChannels # there are binary classes for each channel.
            )
        elif self.sTaskType == 'MPP':
            self.oDecoder = Mpp_Decoder(
                iFfnUnits = input_shape[-1]
            )
        elif self.sTaskType == 'SPP':
            self.oDecoder = Spp_Decoder(
                iFfnUnits = self.iNrOfQuantiles * 3 # due to quantiles. 3 means one-hot categories of signs {positive, negative and zero}
            )
        elif self.sTaskType == 'RPP':
            self.oDecoder = Rpp_Decoder(
                iFfnUnits = self.iNrOfQuantiles * (self.iNrOfChannels + 1) # +1 is because 0 rank is assigned for the positions of special tokens.
            )
                                        
            

        
    def call(self, x):

        x = self.ce(x) + self.pe(x) + self.se(x)

        for oEncoder in self.aTransformerEncoders:
            x = oEncoder(x)
                        
        x = self.oDecoder(x)

        return x
    
    
    