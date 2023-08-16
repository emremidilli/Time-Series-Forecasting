import tensorflow as tf

from tsf_model.layers.pre_processing import LookbackNormalizer, \
    PatchTokenizer, DistributionTokenizer, TrendSeasonalityTokenizer


class PreProcessor(tf.keras.Model):
    '''
        Keras model to pre-process timestep inputs.
    '''
    def __init__(self,
                 iPatchSize,
                 iPoolSizeReduction,
                 iPoolSizeTrend,
                 iNrOfBins,
                 **kwargs):
        super().__init__(**kwargs)

        self.lookback_normalizer = LookbackNormalizer()
        self.patch_tokenizer = PatchTokenizer(iPatchSize)
        self.distribution_tokenizer = DistributionTokenizer(
            iNrOfBins=iNrOfBins,
            fMin=0,
            fMax=1)

        self.trend_seasonality_tokenizer = TrendSeasonalityTokenizer(
            iPoolSizeReduction=iPoolSizeReduction,
            iPoolSizeTrend=iPoolSizeTrend)
        self.lb_fc_concatter = tf.keras.layers.Concatenate(axis=1)

    def call(self, inputs):
        '''
            inputs: tuple of 2 elements.
            Each element is a tf.data.Dataset object.
                1. x_lb: (None, timesteps)
                2. x_fc: (None, timesteps)

            returns tuple of 3 elemements.
                1. dist: (None, timesteps, feature)
                2. tre: (None, timesteps, feature)
                3. sea: (None, timesteps, feature)
        '''

        x_lb, x_fc = inputs

        # normalize
        x_fc = self.lookback_normalizer((x_lb, x_fc))
        x_lb = self.lookback_normalizer((x_lb, x_lb))

        # tokenize
        x_lb = self.patch_tokenizer(x_lb)
        x_fc = self.patch_tokenizer(x_fc)

        x_lb_dist = self.distribution_tokenizer(x_lb)
        x_fc_dist = self.distribution_tokenizer(x_fc)

        x_lb_tre, x_lb_sea = self.trend_seasonality_tokenizer(x_lb)
        x_fc_tre, x_fc_sea = self.trend_seasonality_tokenizer(x_fc)

        # normalize saesonality
        x_lb_sea = self.lookback_normalizer((x_lb_sea, x_lb_sea))
        x_fc_sea = self.lookback_normalizer((x_lb_sea, x_fc_sea))

        dist = self.lb_fc_concatter((x_lb_dist, x_fc_dist))
        tre = self.lb_fc_concatter((x_lb_tre, x_fc_tre))
        sea = self.lb_fc_concatter((x_lb_sea, x_fc_sea))

        return (dist, tre, sea)
