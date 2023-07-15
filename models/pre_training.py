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
                 fMskRate,
                 fMskScalar,
                 iNrOfBins,
                 iNrOfPatches,
                 **kwargs):
        super().__init__(**kwargs)

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


        self.decoder_dist = MppDecoder(iNrOfBins, iNrOfPatches)
        self.decoder_tre = MppDecoder(iPatchSize, iNrOfPatches)
        self.decoder_sea = MppDecoder(iPatchSize, iNrOfPatches)
 

    def call(self, inputs):
        '''
        input: tuple of 6 elements
            x_lb_dist: (None, timesteps, feature)
            x_lb_tre: (None, timesteps, feature)
            x_lb_sea: (None, timesteps, feature)
            x_fc_dist: (None, timesteps, feature)
            x_fc_tre: (None, timesteps, feature)
            x_fc_sea: (None, timesteps, feature)
        '''
        
        x_lb_dist,x_lb_tre, x_lb_sea,x_fc_dist, x_fc_tre , x_fc_sea = inputs

        
        
        # mask
        x_lb_dist_msk, x_lb_tre_msk, x_lb_sea_msk = self.patch_masker((x_lb_dist, x_lb_tre, x_lb_sea))
        x_fc_dist_msk, x_fc_tre_msk, x_fc_sea_msk = self.patch_masker((x_fc_dist,x_fc_tre , x_fc_sea))


        x_dist_msk = self.lookback_forecast_concatter([x_lb_dist_msk, x_fc_dist_msk])
        x_tre_msk = self.lookback_forecast_concatter([x_lb_tre_msk, x_fc_tre_msk])
        x_sea_msk = self.lookback_forecast_concatter([x_lb_sea_msk, x_fc_sea_msk])
        

        # shift
        x_fc_dist_sft = self.patch_shifter(x_fc_dist)
        x_fc_tre_sft = self.patch_shifter(x_fc_tre)
        x_fc_sea_sft = self.patch_shifter(x_fc_sea)

        x_cont_temp = self.encoder_representation((x_dist_msk, x_tre_msk, x_sea_msk))


        y_dist = self.decoder_dist(x_cont_temp)
        y_tre = self.decoder_tre(x_cont_temp)
        y_sea = self.decoder_sea(x_cont_temp)


        
        return (y_dist, y_tre, y_sea)