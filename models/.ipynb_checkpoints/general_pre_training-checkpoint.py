import tensorflow as tf

import sys
sys.path.append( '../')

from layers.general_pre_training.channel_embedding import Channel_Embedding

from layers.general_pre_training.segment_embedding import Segment_Embedding

from layers.general_pre_training.position_embedding import Position_Embedding

from layers.general_pre_training.transformer_encoder import Transformer_Encoder

from layers.general_pre_training.npp_decoder import Npp_Decoder

from layers.general_pre_training.mpp_decoder import Mpp_Decoder

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import CSVLogger

from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import AUC, MeanAbsoluteError

import os

class stopAtThreshold(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        if self.model.sTaskType == 'NPP':
            if logs.get('AUC') >= self.model.STOP_METRIC:
                self.model.stop_training = True
        elif self.model.sTaskType == 'MPP':
            if logs.get('MAE') <= self.model.STOP_METRIC:
                self.model.stop_training = True

class general_pre_training(tf.keras.Model):
    
    def __init__(self, iNrOfEncoderBlocks,iNrOfHeads,fDropoutRate, iEncoderFfnUnits,iEmbeddingDims, sTaskType = None, **kwargs):
        super().__init__(**kwargs)
        
        
        self.iNrOfEncoderBlocks = iNrOfEncoderBlocks
        self.iNrOfHeads = iNrOfHeads
        self.fDropoutRate = fDropoutRate
        self.iEncoderFfnUnits = iEncoderFfnUnits
        self.iEmbeddingDims = iEmbeddingDims
        
        
        self.sTaskType = sTaskType
        self.iNrOfChannels = 3
        self.iNrOfQuantiles = 3

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
                iFfnUnits = iEncoderFfnUnits
            )
            self.aTransformerEncoders.append(oToAdd)
            
        self.SetLossAndMetrics()
            
            
        
    def SetLossAndMetrics(self):
        if self.sTaskType == 'NPP':
            self.oLoss = BinaryCrossentropy()
            self.oMetric = AUC(name = 'AUC')
            self.STOP_METRIC = 0.95
        elif self.sTaskType == 'MPP':
            self.oLoss = MeanSquaredError()
            self.oMetric = MeanAbsoluteError(name = 'MAE')
            self.STOP_METRIC = 0.10
    
    def build(self, input_shape):
        if self.sTaskType == 'NPP':
            self.oDecoder = Npp_Decoder(
                iFfnUnits = self.iNrOfChannels # there are binary classes for each channel.
            )
            
        elif self.sTaskType == 'MPP':
            self.oDecoder = Mpp_Decoder(
                iFfnUnits = input_shape[-1]
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
            if ('__decoder' in s.name) or ('general_pre_training_' not in s.name): #every layer until decoder is already common between models.
                break
            else:
                iIndexDecoder = iIndexDecoder + 1
        
        
        aNewWeights = self.get_weights()
        aNewWeights[:iIndexDecoder] = oModelFrom.get_weights()[:iIndexDecoder]
        self.set_weights(aNewWeights)
        
    


    def Train(self, X_train, Y_train, X_validation, Y_validation ,sArtifactsFolder, fLearningRate, fMomentumRate ,iNrOfEpochs, iMiniBatchSize, iPatience):
        sModelArtifactPath = f'{sArtifactsFolder}\\{self.sTaskType}\\'
        os.makedirs(sModelArtifactPath)        
        
        oCsvLogger = CSVLogger(f'{sModelArtifactPath}logs.log', separator=";", append=False)
        oStopAtThreshold = stopAtThreshold()

        self.compile(
            loss = self.oLoss, 
            metrics = self.oMetric,
            optimizer= Adam(
                learning_rate=ExponentialDecay(
                    initial_learning_rate=fLearningRate,
                    decay_steps=100000,
                    decay_rate=0.96
                ),
                beta_1 = fMomentumRate
            )
        )

        self.fit(
            X_train, 
            Y_train, 
            epochs= iNrOfEpochs, 
            batch_size=iMiniBatchSize, 
            verbose=1,
            validation_data = (X_validation, Y_validation),
            validation_batch_size = iMiniBatchSize,
            callbacks = [oCsvLogger, oStopAtThreshold]
        )

        
        self.save_weights(
            sModelArtifactPath,
            save_format ='tf'
        )