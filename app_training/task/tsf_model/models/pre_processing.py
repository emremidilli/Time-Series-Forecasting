import tensorflow as tf

from tsf_model.layers.pre_processing import LookbackNormalizer, \
    PatchTokenizer, DistributionTokenizer, TrendSeasonalityTokenizer, \
    LayerNormalizer, BatchNormalizer, QuantileTokenizer


class BasePreProcessor(tf.keras.Model):
    def __init__(self,
                 iPatchSize,
                 **kwargs):
        super().__init__(**kwargs)

        self.lookback_normalizer = LookbackNormalizer()
        self.patch_tokenizer = PatchTokenizer(iPatchSize)

    def call(self, inputs):
        '''
            applies lookback normalization and patch tokenization.
            inputs: tuple of 2 elements.
                1. x_lb: (None, timesteps)
                2. x_fc: (None, timesteps)

            returns tuple of 2 elemements.
                1. dist: (None, timesteps, feature)
                2. tre: (None, timesteps, feature)
        '''
        x_lb, x_fc = inputs

        # lookback normalize
        x_fc = self.lookback_normalizer((x_lb, x_fc))
        x_lb = self.lookback_normalizer((x_lb, x_lb))

        # tokenize
        x_lb = self.patch_tokenizer(x_lb)
        x_fc = self.patch_tokenizer(x_fc)

        return (x_lb, x_fc)


class InputPreProcessor(tf.keras.Model):
    '''
        Preprocess for pre-training model.
    '''
    def __init__(self,
                 iPatchSize,
                 iPoolSizeReduction,
                 iPoolSizeTrend,
                 iNrOfBins,
                 **kwargs):
        super().__init__(**kwargs)

        self.base_pre_processor = BasePreProcessor(iPatchSize=iPatchSize)

        self.distribution_tokenizer = DistributionTokenizer(
            iNrOfBins=iNrOfBins,
            fMin=0,
            fMax=1)

        self.trend_seasonality_tokenizer = TrendSeasonalityTokenizer(
            iPoolSizeReduction=iPoolSizeReduction,
            iPoolSizeTrend=iPoolSizeTrend)
        self.lb_fc_concatter = tf.keras.layers.Concatenate(axis=1)

        self.layer_normalizer = LayerNormalizer()

        self.batch_normalizer = BatchNormalizer()

    def call(self, inputs, training=False):
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

        x_lb, x_fc = self.base_pre_processor((x_lb, x_fc))

        x_lb_dist = self.distribution_tokenizer(x_lb)
        x_fc_dist = self.distribution_tokenizer(x_fc)

        x_lb_tre, x_lb_sea = self.trend_seasonality_tokenizer(x_lb)
        x_fc_tre, x_fc_sea = self.trend_seasonality_tokenizer(x_fc)

        dist = self.lb_fc_concatter((x_lb_dist, x_fc_dist))
        tre = self.lb_fc_concatter((x_lb_tre, x_fc_tre))
        sea = self.lb_fc_concatter((x_lb_sea, x_fc_sea))

        y = (dist, tre, sea)
        y = self.layer_normalizer(y)

        return y


class TargetPreProcessor(tf.keras.Model):
    '''Preprocess to prouce target features.'''
    def __init__(self, iPatchSize, quantiles, **kwargs):
        super().__init__(**kwargs)
        self.base_pre_processor = BasePreProcessor(iPatchSize=iPatchSize)

        self.quantile_tokenizer = QuantileTokenizer(quantiles=quantiles)

    def call(self, inputs):
        '''
        input:
            1. x_lb: (None, timesteps)
            2. x_fc: (None, timesteps)
        returns:
            1. qntl: (None, timesteps, features)
        '''
        x_lb, x_fc = inputs

        _, x_fc = self.base_pre_processor((x_lb, x_fc))

        qntl = self.quantile_tokenizer(x_fc)

        return qntl
