import tensorflow as tf

import sys
sys.path.append( '../')


from layers.general_pre_training.mpp_decoder import Mpp_Decoder

from layers.general_pre_training.representation import Representation

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau


from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import AUC, MeanAbsoluteError

import os



class Tokenize_Distribution(tf.keras.layers.Layer):
    
    
    def __init__(self, num_bins, fMin, fMax, **kwargs):
        super().__init__(**kwargs)
        
        self.num_bins = num_bins
        
    '''
        inputs: tuple of 3 elements:
            1. x - original input (None, nr_of_patches, patch_size)
            2. fMin - minimum boundary of bins
            3. fMax - maximum boundary of bins
        
        returns: (None, nr_of_patches, num_bins)
    '''    
    def call(inputs):
        
        x, fMin, fMax = inputs
        
        bin_boundaries=tf.linspace(
            start = fMin, 
            stop = fMax, 
            num = self.num_bins
        )
        
        oDiscritizer = tf.keras.layers.Discretization(bin_boundaries = bin_boundaries)
        
        
        # iNrOfPatches = inputs.shape[1]
        
        y = oDiscritizer(x)
        
        return y
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        

class MaskedAutoEncoder(tf.keras.Model):
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
                
        self.oGPreT = Representation(
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
                    print('Stopping because threshold is achived succesfully...')
                            
            elif self.sMode == 'max':
                if logs.get(self.sMonitor) >= self.fThreshold:
                    self.model.stop_training = True
                    print('Stopping because threshold is achived succesfully...')
                                    
                
    sArtifactsFolder = f'{sArtifactsFolder}/'
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
