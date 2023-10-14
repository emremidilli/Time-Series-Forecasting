import tensorflow as tf

from layers import LookbackNormalizer, PatchTokenizer, \
    DistributionTokenizer, TrendSeasonalityTokenizer


class BasePreProcessor(tf.keras.Model):
    '''
    Base pre-processor that performs lookback normalization
    and patch tokenization.
    '''
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
            1. x_lb: (None, timesteps, feature)
            2. x_fc: (None, timesteps, feature)
        '''
        x_lb, x_fc = inputs

        if x_fc is not None:
            x_fc = self.lookback_normalizer((x_lb, x_fc))

        x_lb = self.lookback_normalizer((x_lb, x_lb))

        x_lb = self.patch_tokenizer(x_lb)
        if x_fc is not None:
            x_fc = self.patch_tokenizer(x_fc)

        return (x_lb, x_fc)


class InputPreProcessorPT(tf.keras.Model):
    '''
    Preprocess for input of pre-training.
    Distribution, trend and seasonility components are prepared.
    Trend, seasonality components and timestamp features are normalized.
    Distribution is not normalized since its data is within 0 and 1.
    '''
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

        self.tre_normalizer = tf.keras.layers.Normalization(axis=None)
        self.sea_normalizer = tf.keras.layers.Normalization(axis=None)
        self.ts_normalizer = tf.keras.layers.Normalization(axis=1)

    def adapt(self, inputs):
        '''
        Adapts the mean and standard deviaiton of
        trend, seasonality and timestamp features.
        inputs: tuple of 3 elements.
        Each element is a tf.data.Dataset object.
            1. x_lb: (None, timesteps)
            2. x_fc: (None, timesteps)
            3. x_ts: (None, features)
        '''
        x_lb, x_fc, x_ts = inputs

        x_lb, x_fc = self.base_pre_processor((x_lb, x_fc))

        x_lb_tre, x_lb_sea = self.trend_seasonality_tokenizer(x_lb)

        x_fc_tre, x_fc_sea = self.trend_seasonality_tokenizer(x_fc)

        x_tre = self.lb_fc_concatter((x_lb_tre, x_fc_tre))
        x_sea = self.lb_fc_concatter((x_lb_sea, x_fc_sea))
        x_ts = tf.cast(x_ts, dtype=tf.float32)

        self.tre_normalizer.adapt(x_tre)
        self.sea_normalizer.adapt(x_sea)
        self.ts_normalizer.adapt(x_ts)

    def call(self, inputs, training=False):
        '''
        inputs: tuple of 3 elements.
        Each element is a tf.data.Dataset object.
            1. x_lb: (None, timesteps)
            2. x_fc: (None, timesteps)
            3. x_ts: (None, features)

        returns tuple of 4 elemements.
            1. y_dist: (None, timesteps, features)
            2. y_tre: (None, timesteps, features)
            3. y_sea: (None, timesteps, features)
            4. y_ts: (None, features)
        '''

        x_lb, x_fc, x_ts = inputs

        x_lb, x_fc = self.base_pre_processor((x_lb, x_fc))

        x_lb_dist = self.distribution_tokenizer(x_lb)
        x_lb_tre, x_lb_sea = self.trend_seasonality_tokenizer(x_lb)

        x_fc_dist = self.distribution_tokenizer(x_fc)
        x_fc_tre, x_fc_sea = self.trend_seasonality_tokenizer(x_fc)

        x_dist = self.lb_fc_concatter((x_lb_dist, x_fc_dist))
        x_tre = self.lb_fc_concatter((x_lb_tre, x_fc_tre))
        x_sea = self.lb_fc_concatter((x_lb_sea, x_fc_sea))
        x_ts = tf.cast(x_ts, dtype=tf.float32)

        y_dist = x_dist
        y_tre = self.tre_normalizer(x_tre)
        y_sea = self.sea_normalizer(x_sea)
        y_ts = self.ts_normalizer(x_ts)

        return (y_dist, y_tre, y_sea, y_ts)


class InputPreProcessorFT(tf.keras.Model):
    '''
    Preprocess for input of fine-tuning models.
    Distribution, trend and seasonility components are prepared.
    Trend, seasonality components and timestamp features are normalized.
    Distribution is not normalized since its data is within 0 and 1.
    Unlikely to pre-training, only lookback patches are normalized.
    Forecast patches are replaced with mask_scalar value.
    '''
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

        self.tre_normalizer = tf.keras.layers.Normalization(axis=None)
        self.sea_normalizer = tf.keras.layers.Normalization(axis=None)
        self.ts_normalizer = tf.keras.layers.Normalization(axis=1)

    def adapt(self, inputs):
        '''
        Adapts the mean and standard deviaiton of
        trend, seasonality and timestamp features.
        inputs: tuple of 2 elements.
        Each element is a tf.data.Dataset object.
            1. x_lb: (None, timesteps)
            2. x_ts: (None, feature) timestamp features
        '''
        (x_lb, x_ts) = inputs

        x_ts = tf.cast(x_ts, dtype=tf.float32)

        x_lb, _ = self.base_pre_processor((x_lb, None))

        x_lb_tre, x_lb_sea = self.trend_seasonality_tokenizer(x_lb)

        self.tre_normalizer.adapt(x_lb_tre)
        self.sea_normalizer.adapt(x_lb_sea)
        self.ts_normalizer.adapt(x_ts)

    def call(self, inputs):
        '''
        inputs: tuple of 2 elements.
        Each element is a tf.data.Dataset object.
            1. x_lb: (None, timesteps)
            2. x_ts: (None, feature) timestamp features

        returns tuple of 3 elemements.
            1. dist: (None, timesteps, feature)
            2. tre: (None, timesteps, feature)
            3. sea: (None, timesteps, feature)
        '''
        (x_lb, x_ts) = inputs

        x_ts = tf.cast(x_ts, dtype=tf.float32)

        x_lb, _ = self.base_pre_processor((x_lb, None))

        x_lb_dist = self.distribution_tokenizer(x_lb)
        x_lb_tre, x_lb_sea = self.trend_seasonality_tokenizer(x_lb)

        x_lb_tre = self.tre_normalizer(x_lb_tre)
        x_lb_sea = self.sea_normalizer(x_lb_sea)
        x_ts = self.ts_normalizer(x_ts)

        x_fc_dist = tf.zeros_like(x_lb_dist) + self.mask_scalar
        x_fc_tre = tf.zeros_like(x_lb_tre) + self.mask_scalar
        x_fc_sea = tf.zeros_like(x_lb_sea) + self.mask_scalar

        x_fc_dist = x_fc_dist[:, :self.forecast_patches_to_mask]
        x_fc_tre = x_fc_tre[:, :self.forecast_patches_to_mask]
        x_fc_sea = x_fc_sea[:, :self.forecast_patches_to_mask]

        y_dist = self.lb_fc_concatter((x_lb_dist, x_fc_dist))
        y_tre = self.lb_fc_concatter((x_lb_tre, x_fc_tre))
        y_sea = self.lb_fc_concatter((x_lb_sea, x_fc_sea))

        y_ts = x_ts

        return (y_dist, y_tre, y_sea, y_ts)


class TargetPreProcessor(tf.keras.Model):
    '''Preprocess to produce target features.'''
    def __init__(self, patch_size, begin_scalar, end_scalar, **kwargs):
        super().__init__(**kwargs)
        self.base_pre_processor = BasePreProcessor(patch_size=patch_size)
        self.reshaper = tf.keras.layers.Reshape((-1, 1))
        self.begin_scalar = begin_scalar
        self.end_scalar = end_scalar

    def call(self, inputs):
        '''
        input:
        x_lb: (None, timesteps)
        x_fc: (None, timesteps)
        y: (None, timesteps, 1)
        y_shifted: (None, timesteps, 1)
        '''
        x_lb, x_fc = inputs

        _, x_fc = self.base_pre_processor((x_lb, x_fc))

        x_fc = self.reshaper(x_fc)

        z = tf.roll(x_fc, shift=1, axis=1)
        beg_token = tf.zeros_like(z) + self.begin_scalar
        end_token = tf.zeros_like(z) + self.end_scalar
        z = tf.keras.layers.Concatenate(axis=1)(
            [
                tf.expand_dims(beg_token[:, 0], axis=1),
                x_fc,
                tf.expand_dims(end_token[:, 0], axis=1),
            ])
        y = z[:, 1:]
        y_shifted = z[:, :-1]
        return (y, y_shifted)
