import sys
sys.path.append( './')

import tensorflow as tf

from layers.pre_processing import *
from layers.general_pre_training import *


class PreTraining(tf.keras.Model):
    '''
        Keras model for pre-training purpose.
    '''

    def __init__(self,
                 iNrOfEncoderBlocks,
                 iNrOfHeads, 
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


        self.lookback_normalizer = LookbackNormalizer()
        self.patch_tokenizer = PatchTokenizer(iPatchSize)
        self.distribution_tokenizer = DistributionTokenizer(
            iNrOfBins=iNrOfBins,
            fMin=0,
            fMax=1
            )
        
        self.trend_seasonality_tokenizer = TrendSeasonalityTokenizer(int(fPatchSampleRate * iPatchSize))

        self.patch_masker = PatchMasker(fMaskingRate=fMskRate, fMskScalar=fMskScalar)

        self.patch_shifter = PatchShifter()
        
        self.encoder_representation = Representation(
            iNrOfEncoderBlocks,
            iNrOfHeads,
            fDropoutRate, 
            iEncoderFfnUnits,
            iEmbeddingDims
        )


        self.lookback_forecast_concatter = tf.keras.layers.Concatenate(axis = 1)

        

    def call(self, x):
        '''
        input: tuple of 2 elements
            x_lb: (None, timesteps)
            x_fc: (None, timesteps)
        '''
        
        x_lb, x_fc = x

        # normalize
        x_fc = self.lookback_normalizer((x_lb,x_fc))
        x_lb = self.lookback_normalizer((x_lb,x_lb))
        
        # tokenize
        x_lb = self.patch_tokenizer(x_lb)
        x_fc = self.patch_tokenizer(x_fc)

        x_lb_dist = self.distribution_tokenizer(x_lb)
        x_fc_dist = self.distribution_tokenizer(x_fc)
        
        x_lb_tre,x_lb_sea  = self.trend_seasonality_tokenizer(x_lb)
        x_fc_tre,x_fc_sea  = self.trend_seasonality_tokenizer(x_fc)
        
        # normalize saesonality
        x_lb_sea = self.lookback_normalizer((x_lb_sea,x_lb_sea))
        x_fc_sea = self.lookback_normalizer((x_fc_sea,x_fc_sea))

        
        # mask
        x_lb_dist_msk = self.patch_masker(x_lb_dist)
        x_fc_dist_msk = self.patch_masker(x_fc_dist)

        x_lb_tre_msk = self.patch_masker(x_lb_tre)
        x_fc_tre_msk = self.patch_masker(x_fc_tre)   

        x_lb_sea_msk = self.patch_masker(x_lb_sea)
        x_fc_sea_msk = self.patch_masker(x_fc_sea)


        x_dist_msk = self.lookback_forecast_concatter([x_lb_dist_msk, x_fc_dist_msk])
        x_tre_msk = self.lookback_forecast_concatter([x_lb_tre_msk, x_fc_tre_msk])
        x_sea_msk = self.lookback_forecast_concatter([x_lb_sea_msk, x_fc_sea_msk])
        

        # shift
        x_fc_dist_sft = self.patch_shifter(x_fc_dist)
        x_fc_tre_sft = self.patch_shifter(x_fc_tre)
        x_fc_sea_sft = self.patch_shifter(x_fc_sea)

        x_cont_temp = self.encoder_representation((x_dist_msk, x_tre_msk, x_sea_msk))


        



        
        return x_lb