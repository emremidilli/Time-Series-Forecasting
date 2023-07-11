import tensorflow as tf

import sys
sys.path.append( '/home/yunusemre/Time-Series-Forecasting/')

from layers.pre_processing.preprocessor import *
from layers.general_pre_training.mpp_decoder import Mpp_Decoder

from layers.general_pre_training.representation import Representation

from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau


from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.metrics import AUC, MeanAbsoluteError

import os


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
                 iPatchSize,
                 fPatchSampleRate,
                 fMskRate,
                 fMskScalar,
                 iNrOfBins,
                 **kwargs):
        super().__init__(**kwargs)


        self.oLookbackNormalizer = LookbackNormalizer()
        self.oPatchTokenizer = PatchTokenizer(iPatchSize)
        self.oDistTokenizer = DistributionTokenizer(
            iNrOfBins=iNrOfBins,
            fMin=0,
            fMax=1 #relaxation can be applied. (eg. tredinding series)
            )
        
        self.oTsTokenizer = TrendSeasonalityTokenizer(int(fPatchSampleRate * iPatchSize))

        self.oPatchMasker = PatchMasker(fMaskingRate=fMskRate, fMskScalar=fMskScalar)

        self.oPatchShifter = PatchShifter()
        
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
        
        x_lb, x_fc = x

        # normalize
        x_fc = self.oLookbackNormalizer((x_lb,x_fc))
        x_lb = self.oLookbackNormalizer((x_lb,x_lb))
        
        # tokenize
        x_lb = self.oPatchTokenizer(x_lb)
        x_fc = self.oPatchTokenizer(x_fc)

        x_lb_dist = self.oDistTokenizer(x_lb)
        x_fc_dist = self.oDistTokenizer(x_fc)
        
        x_lb_tre,x_lb_sea  = self.oTsTokenizer(x_lb)
        x_fc_tre,x_fc_sea  = self.oTsTokenizer(x_fc)
        

        # mask
        x_lb_dist_msk = self.oPatchMasker(x_lb_dist)
        x_fc_dist_msk = self.oPatchMasker(x_fc_dist)

        x_lb_tre_msk = self.oPatchMasker(x_lb_tre)
        x_fc_tre_msk = self.oPatchMasker(x_fc_tre)   

        x_lb_sea_msk = self.oPatchMasker(x_lb_sea)
        x_fc_sea_msk = self.oPatchMasker(x_fc_sea)   


        # shift
        x_fc_dist_sft = self.oPatchShifter(x_fc_dist)
        x_fc_tre_sft = self.oPatchShifter(x_fc_tre)
        x_fc_sea_sft = self.oPatchShifter(x_fc_sea)


       
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


