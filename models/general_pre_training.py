import tensorflow as tf

import sys
sys.path.append( '../')

from layers.general_pre_training.channel_embedding import Channel_Embedding

from layers.general_pre_training.segment_embedding import Segment_Embedding

from layers.general_pre_training.position_embedding import Position_Embedding

from layers.general_pre_training.transformer_encoder import Transformer_Encoder

from layers.general_pre_training.npp_decoder import Npp_Decoder

from layers.general_pre_training.mpp_decoder import Mpp_Decoder

from layers.general_pre_training.rpp_decoder import Rpp_Decoder

from layers.general_pre_training.spp_decoder import Spp_Decoder



from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau


from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import AUC, MeanAbsoluteError

import os



class GPReT(tf.keras.layers.Layer):
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

    

    

class masked_patch_prediction(tf.keras.Model):
    def __init__(self,
                 iNrOfChannels, 
                 iNrOfQuantiles,
                 iNrOfLookbackPatches,
                 iNrOfForecastPatches,  
                 iNrOfEncoderBlocks,
                 iNrOfHeads, 
                 iContextSize , 
                 fDropoutRate, 
                 iEncoderFfnUnits,
                 iEmbeddingDims, 
                 **kwargs):
        super().__init__(**kwargs)
                
        self.oGPreT = GPReT(
            iNrOfChannels , iNrOfQuantiles,iNrOfLookbackPatches, iNrOfForecastPatches  ,
            iNrOfEncoderBlocks,iNrOfHeads,fDropoutRate, iEncoderFfnUnits,iEmbeddingDims
        )

            
        self.oDecoder = Mpp_Decoder(
            iFfnUnits = iContextSize
        )
        
        self.oLoss = MeanSquaredError(name = 'mse')
        self.oMetrics = MeanAbsoluteError(name = 'mae')
        self.sMode = 'min'
        self.fThreshold = 0.05


    def call(self, x):
        
        x = self.oGPreT(x)
        
        x = self.oDecoder(x)
        
        return x
    
    


    
def Train(
    oModel, 
    X_train, 
    Y_train, 
    sArtifactsFolder,
    fLearningRate,
    fMomentumRate ,
    iNrOfEpochs, 
    iMiniBatchSize
):

    class stopAtThreshold(tf.keras.callbacks.Callback):
        
        def __init__(self, sMonitor, fThreshold, sMode):
            self.sMonitor = sMonitor
            self.fThreshold = fThreshold
            self.sMode = sMode
            
            
        
        def on_batch_end(self, batch, logs={}):
            
            print(logs)
            if self.sMode == 'min':
                if logs.get(self.sMonitor) <= self.fThreshold:
                    self.model.stop_training = True
                            
            elif self.sMode == 'max':
                if logs.get(self.sMonitor) >= self.fThreshold:
                    self.model.stop_training = True
                                    
                
    sArtifactsFolder = f'{sArtifactsFolder}\\'
    os.makedirs(sArtifactsFolder)

    oCsvLogger = CSVLogger(f'{sArtifactsFolder}logs.log', separator=";", append=False)
    oStopAtThreshold = stopAtThreshold(
        sMonitor = oModel.oMetrics.name, 
        fThreshold = oModel.fThreshold, 
        sMode = oModel.sMode
    )

    oReduceLr = ReduceLROnPlateau(
        monitor='loss', 
        factor=0.2, 
        patience= 3, 
        min_lr=0.0001
    )

    oEarlyStopping = EarlyStopping(
        monitor='loss', 
        patience=5, 
        restore_best_weights=True
    )

    oModel.compile(
        loss = oModel.oLoss,
        metrics = oModel.oMetrics,
        optimizer= Adam(
            learning_rate=ExponentialDecay(
                initial_learning_rate=fLearningRate,
                decay_steps=100000,
                decay_rate=0.96
            ),
            beta_1 = fMomentumRate
        )
    )

    oModel.fit(
        X_train, 
        Y_train, 
        epochs= iNrOfEpochs, 
        batch_size=iMiniBatchSize, 
        verbose=1,
        callbacks = [oCsvLogger, oStopAtThreshold, oReduceLr, oEarlyStopping]
    )

    oModel.save(
        sArtifactsFolder, 
        overwrite = True,
        save_format = 'tf')
