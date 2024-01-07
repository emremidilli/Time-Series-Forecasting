import tensorflow as tf

from layers import PatchTokenizer, TrendSeasonalityTokenizer


class InputPreProcessorPT(tf.keras.Model):
    '''
    Preprocess for input of pre-training.
    Time series decomposition is applied.
    The components are normalized.
    '''
    def __init__(self,
                 patch_size,
                 pool_size_trend,
                 nr_of_covariates,
                 sigma,
                 **kwargs):
        super().__init__(**kwargs)

        self.patch_tokenizer = PatchTokenizer(
            patch_size=patch_size,
            nr_of_covariates=nr_of_covariates)

        self.trend_seasonality_tokenizer = TrendSeasonalityTokenizer(
            pool_size_trend=pool_size_trend,
            nr_of_covariates=nr_of_covariates,
            sigma=sigma)
        self.lb_fc_concatter = tf.keras.layers.Concatenate(axis=1)

        self.tre_normalizer = tf.keras.layers.Normalization(axis=None)
        self.sea_normalizer = tf.keras.layers.Normalization(axis=None)
        self.res_normalizer = tf.keras.layers.Normalization(axis=None)
        self.ts_normalizer = tf.keras.layers.Normalization(axis=1)

        self.tre_denormalizer = \
            tf.keras.layers.Normalization(axis=None, invert=True)
        self.sea_denormalizer = \
            tf.keras.layers.Normalization(axis=None, invert=True)
        self.res_denormalizer = \
            tf.keras.layers.Normalization(axis=None, invert=True)
        self.ts_denormalizer = \
            tf.keras.layers.Normalization(axis=1, invert=True)

    def adapt(self, inputs):
        '''
        Adapts the mean and standard deviation of components.
        args:
            inputs (tuple) - tuple of 3 elements.
                Each element is a tf.data.Dataset object.
                1. x_lb: (None, timesteps, covariates)
                2. x_fc: (None, timesteps, covariates)
                3. x_ts: (None, features)
        '''
        x_lb, x_fc, x_ts = inputs

        x_lb = self.patch_tokenizer(x_lb)
        x_fc = self.patch_tokenizer(x_fc)

        x_lb_tre, x_lb_sea, x_lb_res = self.trend_seasonality_tokenizer(x_lb)

        x_fc_tre, x_fc_sea, x_fc_res = self.trend_seasonality_tokenizer(x_fc)

        x_tre = self.lb_fc_concatter((x_lb_tre, x_fc_tre))
        x_sea = self.lb_fc_concatter((x_lb_sea, x_fc_sea))
        x_res = self.lb_fc_concatter((x_lb_res, x_fc_res))
        x_ts = tf.cast(x_ts, dtype=tf.float32)

        self.tre_normalizer.adapt(x_tre)
        self.sea_normalizer.adapt(x_sea)
        self.res_normalizer.adapt(x_res)
        self.ts_normalizer.adapt(x_ts)

        self.tre_denormalizer.adapt(x_tre)
        self.sea_denormalizer.adapt(x_sea)
        self.res_denormalizer.adapt(x_res)
        self.ts_denormalizer.adapt(x_ts)

    def call(self, inputs, training=False):
        '''
        args:
            inputs (tuple): tuple of 3 elements.
                Each element is a tf.data.Dataset object.
                1. x_lb: (None, timesteps, covariates)
                2. x_fc: (None, timesteps, covariates)
                3. x_ts: (None, features)

        returns:
            1. y_tre: (None, timesteps, features)
            2. y_sea: (None, timesteps, features)
            3. y_res: (None, timesteps, features)
            4. y_ts: (None, features)
        '''

        x_lb, x_fc, x_ts = inputs

        x_lb = self.patch_tokenizer(x_lb)
        x_fc = self.patch_tokenizer(x_fc)

        x_lb_tre, x_lb_sea, x_lb_res = self.trend_seasonality_tokenizer(x_lb)

        x_fc_tre, x_fc_sea, x_fc_res = self.trend_seasonality_tokenizer(x_fc)

        x_tre = self.lb_fc_concatter((x_lb_tre, x_fc_tre))
        x_sea = self.lb_fc_concatter((x_lb_sea, x_fc_sea))
        x_res = self.lb_fc_concatter((x_lb_res, x_fc_res))
        x_ts = tf.cast(x_ts, dtype=tf.float32)

        y_tre = self.tre_normalizer(x_tre)
        y_sea = self.sea_normalizer(x_sea)
        y_res = self.res_normalizer(x_res)
        y_ts = self.ts_normalizer(x_ts)

        y_tre = tf.reshape(
            tensor=y_tre,
            shape=(tf.shape(y_tre)[0], tf.shape(y_tre)[1], -1))
        y_sea = tf.reshape(
            tensor=y_sea,
            shape=(tf.shape(y_sea)[0], tf.shape(y_sea)[1], -1))
        y_res = tf.reshape(
            tensor=y_res,
            shape=(tf.shape(y_res)[0], tf.shape(y_res)[1], -1))

        return (y_tre, y_sea, y_res, y_ts)


class InputPreProcessorFT(tf.keras.Model):
    '''
    Preprocess for input of fine-tuning models.
    Trend, seasonility and residual components are prepared.
    The components are normalized.
    Unlikely to pre-training, only lookback patches are normalized.
    Forecast patches are replaced with mask_scalar value.
    '''
    def __init__(self,
                 patch_size,
                 pool_size_trend,
                 nr_of_covariates,
                 forecast_patches_to_mask=None,
                 mask_scalar=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.mask_scalar = mask_scalar
        self.forecast_patches_to_mask = forecast_patches_to_mask

        self.patch_tokenizer = PatchTokenizer(
            patch_size=patch_size,
            nr_of_covariates=nr_of_covariates)

        self.trend_seasonality_tokenizer = TrendSeasonalityTokenizer(
            pool_size_trend=pool_size_trend,
            nr_of_covariates=nr_of_covariates)

        self.lb_fc_concatter = tf.keras.layers.Concatenate(axis=1)

        self.tre_normalizer = tf.keras.layers.Normalization(axis=None)
        self.sea_normalizer = tf.keras.layers.Normalization(axis=None)
        self.res_normalizer = tf.keras.layers.Normalization(axis=None)
        self.ts_normalizer = tf.keras.layers.Normalization(axis=1)

    def adapt(self, inputs):
        '''
        Adapts the mean and standard deviaiton of
        trend, seasonality, residual and timestamp features.
        inputs: tuple of 2 elements.
        Each element is a tf.data.Dataset object.
            1. x_lb: (None, timesteps, covariates)
            2. x_ts: (None, feature) timestamp features
        '''
        (x_lb, x_ts) = inputs

        x_ts = tf.cast(x_ts, dtype=tf.float32)

        x_lb = self.patch_tokenizer(x_lb)

        x_lb_tre, x_lb_sea, x_lb_res = self.trend_seasonality_tokenizer(x_lb)

        self.tre_normalizer.adapt(x_lb_tre)
        self.sea_normalizer.adapt(x_lb_sea)
        self.res_normalizer.adapt(x_lb_res)
        self.ts_normalizer.adapt(x_ts)

    def call(self, inputs):
        '''
        inputs: tuple of 2 elements.
        Each element is a tf.data.Dataset object.
            1. x_lb: (None, timesteps, covariates)
            2. x_ts: (None, feature) timestamp features

        returns tuple of 4 elemements.
            1. y_tre: (None, timesteps, features, covariates)
            2. y_sea: (None, timesteps, features, covariates)
            3. y_res: (None, timesteps, features, covariates)
            4. y_ts: (None, features)
        '''
        (x_lb, x_ts) = inputs

        x_ts = tf.cast(x_ts, dtype=tf.float32)

        x_lb = self.patch_tokenizer(x_lb)

        x_lb_tre, x_lb_sea, x_lb_res = self.trend_seasonality_tokenizer(x_lb)

        x_lb_tre = self.tre_normalizer(x_lb_tre)
        x_lb_sea = self.sea_normalizer(x_lb_sea)
        x_lb_res = self.res_normalizer(x_lb_res)
        x_ts = self.ts_normalizer(x_ts)

        x_fc_tre = tf.zeros_like(x_lb_tre) + self.mask_scalar
        x_fc_sea = tf.zeros_like(x_lb_sea) + self.mask_scalar
        x_fc_res = tf.zeros_like(x_lb_res) + self.mask_scalar

        x_fc_tre = x_fc_tre[:, :self.forecast_patches_to_mask]
        x_fc_sea = x_fc_sea[:, :self.forecast_patches_to_mask]
        x_fc_res = x_fc_res[:, :self.forecast_patches_to_mask]

        y_tre = self.lb_fc_concatter((x_lb_tre, x_fc_tre))
        y_sea = self.lb_fc_concatter((x_lb_sea, x_fc_sea))
        y_res = self.lb_fc_concatter((x_lb_res, x_fc_res))

        y_ts = x_ts

        return (y_tre, y_sea, y_res, y_ts)


class TargetPreProcessor(tf.keras.Model):
    '''Preprocess to produce target features.'''
    def __init__(self, patch_size, begin_scalar, end_scalar, **kwargs):
        super().__init__(**kwargs)
        self.patch_tokenizer = PatchTokenizer(patch_size=patch_size)
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

        x_fc = self.patch_tokenizer(x_fc)

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
