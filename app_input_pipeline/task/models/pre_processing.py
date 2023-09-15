import tensorflow as tf

from layers import LookbackNormalizer, \
    PatchTokenizer, DistributionTokenizer, TrendSeasonalityTokenizer, \
    LayerNormalizer, BatchNormalizer, QuantileTokenizer


class BasePreProcessor(tf.keras.Model):
    def __init__(self,
                 patch_size,
                 **kwargs):
        super().__init__(**kwargs)

        self.lookback_normalizer = LookbackNormalizer()
        self.patch_tokenizer = PatchTokenizer(patch_size)

    def call(self, inputs):
        '''
            applies lookback normalization and patch tokenization.
            inputs: tuple of 2 elements.
                1. x_lb: (None, timesteps)
                2. x_fc: (None, timesteps)
                    if x_fc is None, it is acted as masked.

            returns tuple of 2 elemements.
                1. dist: (None, timesteps, feature)
                2. tre: (None, timesteps, feature)
        '''
        x_lb, x_fc = inputs

        # lookback normalize
        if x_fc is not None:
            x_fc = self.lookback_normalizer((x_lb, x_fc))

        x_lb = self.lookback_normalizer((x_lb, x_lb))

        # tokenize
        x_lb = self.patch_tokenizer(x_lb)
        if x_fc is not None:
            x_fc = self.patch_tokenizer(x_fc)

        return (x_lb, x_fc)


class InputPreProcessorPT(tf.keras.Model):
    '''Preprocess for input of pre-training.'''
    def __init__(self,
                 patch_size,
                 pool_size_reduction,
                 pool_size_trend,
                 nr_of_bins,
                 **kwargs):
        super().__init__(**kwargs)

        self.base_pre_processor = BasePreProcessor(patch_size=patch_size)

        self.distribution_tokenizer = DistributionTokenizer(
            nr_of_bins=nr_of_bins,
            fMin=0,
            fMax=1)

        self.trend_seasonality_tokenizer = TrendSeasonalityTokenizer(
            pool_size_reduction=pool_size_reduction,
            pool_size_trend=pool_size_trend)
        self.lb_fc_concatter = tf.keras.layers.Concatenate(axis=1)

        self.layer_normalizer_lb = LayerNormalizer()

        self.layer_normalizer_fc = LayerNormalizer()

        self.batch_normalizer = BatchNormalizer()

    def call(self, inputs, training=False):
        '''
            inputs: tuple of 3 elements.
            Each element is a tf.data.Dataset object.
                1. x_lb: (None, timesteps)
                2. x_fc: (None, timesteps)
                3. x_ts: (None, features)

            returns tuple of 3 elemements.
                1. dist: (None, timesteps, features)
                2. tre: (None, timesteps, features)
                3. sea: (None, timesteps, features)
        '''

        x_lb, x_fc, x_ts = inputs

        x_lb, x_fc = self.base_pre_processor((x_lb, x_fc))

        x_lb_dist = self.distribution_tokenizer(x_lb)
        x_lb_tre, x_lb_sea = self.trend_seasonality_tokenizer(x_lb)

        (x_lb_dist, x_lb_tre, x_lb_sea) = self.layer_normalizer_lb(
            (x_lb_dist, x_lb_tre, x_lb_sea))

        x_fc_dist = self.distribution_tokenizer(x_fc)
        x_fc_tre, x_fc_sea = self.trend_seasonality_tokenizer(x_fc)

        (x_fc_dist, x_fc_tre, x_fc_sea) = self.layer_normalizer_fc(
            (x_fc_dist, x_fc_tre, x_fc_sea))

        dist = self.lb_fc_concatter((x_lb_dist, x_fc_dist))
        tre = self.lb_fc_concatter((x_lb_tre, x_fc_tre))
        sea = self.lb_fc_concatter((x_lb_sea, x_fc_sea))

        y_ts = self.batch_normalizer(x_ts)

        return (dist, tre, sea, y_ts)


class InputPreProcessorFT(tf.keras.Model):
    '''Preprocess for input of fine-tuning models.'''
    def __init__(self,
                 patch_size,
                 pool_size_reduction,
                 pool_size_trend,
                 nr_of_bins,
                 forecast_patches_to_mask=None,
                 mask_scalar=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.mask_scalar = mask_scalar
        self.forecast_patches_to_mask = forecast_patches_to_mask

        self.base_pre_processor = BasePreProcessor(patch_size=patch_size)

        self.distribution_tokenizer = DistributionTokenizer(
            nr_of_bins=nr_of_bins,
            fMin=0,
            fMax=1)

        self.trend_seasonality_tokenizer = TrendSeasonalityTokenizer(
            pool_size_reduction=pool_size_reduction,
            pool_size_trend=pool_size_trend)
        self.lb_fc_concatter = tf.keras.layers.Concatenate(axis=1)

        self.layer_normalizer_lb = LayerNormalizer()

        self.batch_normalizer = BatchNormalizer()

    def call(self, inputs, training=False):
        '''
            inputs: tuple of 2 elements.
            Each element is a tf.data.Dataset object.
                1. x_lb: (None, timesteps)
                2. x_ts: (None, featires) timestamp features

            returns tuple of 3 elemements.
                1. dist: (None, timesteps, feature)
                2. tre: (None, timesteps, feature)
                3. sea: (None, timesteps, feature)
        '''

        (x_lb, x_ts) = inputs

        x_lb, _ = self.base_pre_processor((x_lb, None))

        x_lb_dist = self.distribution_tokenizer(x_lb)
        x_lb_tre, x_lb_sea = self.trend_seasonality_tokenizer(x_lb)

        (x_lb_dist, x_lb_tre, x_lb_sea) = self.layer_normalizer_lb(
            (x_lb_dist, x_lb_tre, x_lb_sea))

        x_fc_dist = tf.zeros_like(x_lb_dist) + self.mask_scalar
        x_fc_tre = tf.zeros_like(x_lb_tre) + self.mask_scalar
        x_fc_sea = tf.zeros_like(x_lb_sea) + self.mask_scalar

        x_fc_dist = x_fc_dist[:, :self.forecast_patches_to_mask]
        x_fc_tre = x_fc_tre[:, :self.forecast_patches_to_mask]
        x_fc_sea = x_fc_sea[:, :self.forecast_patches_to_mask]

        dist = self.lb_fc_concatter((x_lb_dist, x_fc_dist))
        tre = self.lb_fc_concatter((x_lb_tre, x_fc_tre))
        sea = self.lb_fc_concatter((x_lb_sea, x_fc_sea))

        y_ts = self.batch_normalizer(x_ts)

        return (dist, tre, sea, y_ts)


class TargetPreProcessor(tf.keras.Model):
    '''Preprocess to prouce target features.'''
    def __init__(self, patch_size, quantiles, **kwargs):
        super().__init__(**kwargs)
        self.base_pre_processor = BasePreProcessor(patch_size=patch_size)

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
