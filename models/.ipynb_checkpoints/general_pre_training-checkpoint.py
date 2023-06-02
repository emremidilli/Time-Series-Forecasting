import tensorflow as tf

import sys
sys.path.append( '../')

from layers.general_pre_training.channel_embedding import Channel_Embedding

from layers.general_pre_training.segment_embedding import Segment_Embedding

from layers.general_pre_training.position_embedding import Position_Embedding

from layers.general_pre_training.transformer_embedding import Transformer_Encoder

from layers.general_pre_training.npp_decoder import Npp_Decoder

from layers.general_pre_training.mpp_decoder import Mpp_Decoder

from layers.general_pre_training.spp_decoder import Spp_Decoder

from layers.general_pre_training.rpp_decoder import Rpp_Decoder

import preprocessing.constants as c


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class general_pre_training(tf.keras.Model):
    
    def __init__(self, sTaskType = None, **kwargs):
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
                       
        if self.sTaskType != None:
            x = self.oDecoder(x)

        return x
    
    

    
    def TransferLearningForEncoder(self, oModelFrom):
        iIndexDecoder = 0
        for s in oModelFrom.weights: 
            if '__decoder' in s.name: #every layer until decoder is already common between models.
                break
            else:
                iIndexDecoder = iIndexDecoder + 1
        
        
        aNewWeights = self.get_weights()
        aNewWeights[:iIndexDecoder] = oModelFrom.get_weights()[:iIndexDecoder]
        self.set_weights(aNewWeights)


    def Train(self, X_train, Y_train, sArtifactsFolder, fLearningRate, iNrOfEpochs, iBatchSize ,oLoss, oMetrics):    
        self.compile(
            loss = oLoss, 
            metrics = oMetrics,
            optimizer= Adam(
                learning_rate=ExponentialDecay(
                    initial_learning_rate=fLearningRate,
                    decay_steps=10**2,
                    decay_rate=0.9
                )
            )
        )

        self.fit(
            X_train, 
            Y_train, 
            epochs= iNrOfEpochs, 
            batch_size=iBatchSize, 
            verbose=1
        )

        
        sModelArtifactPath = f'{sArtifactsFolder}\\{self.sTaskType}\\'
        self.save_weights(
            sModelArtifactPath,
            save_format ='tf'
        )