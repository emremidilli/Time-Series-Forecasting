
import tensorflow as tf
from layers.pre_processing import *

class PreProcessor(tf.keras.Model):

    '''
        Keras model to pre-process timestep inputs.
    '''

    def __init__(self,
                 iPatchSize,
                 fPatchSampleRate,
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

    def concat_lb_fc(self, inputs):

        x_lb, x_fc = inputs
        x = tf.keras.layers.Concatenate(axis = 1)([x_lb, x_fc])

        return x



    def call(self, inputs):
        '''
            inputs: tuple of 2 elements.
                1. x_lb: (None, timesteps)
                2. x_fc: (None, timesteps)
                
            returns tuple of 8 elemements.
                1. x_lb: (None, timesteps, feature)
                2. x_lb_dist: (None, timesteps, feature)
                3. x_lb_tre: (None, timesteps, feature)
                4. x_lb_sea: (None, timesteps, feature)
                5. x_fc: (None, timesteps, feature)
                6. x_fc_dist: (None, timesteps, feature)
                7. x_fc_tre: (None, timesteps, feature)
                8. x_fc_sea: (None, timesteps, feature)
        '''

        x_lb , x_fc = inputs

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
        x_fc_sea = self.lookback_normalizer((x_lb_sea,x_fc_sea))




        return (x_lb, x_lb_dist, x_lb_tre, x_lb_sea, x_fc, x_fc_dist, x_fc_tre, x_fc_sea)

            
            