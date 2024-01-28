import tensorflow as tf

from layers import TrendSeasonalityTokenizer


class InputPreProcessorPT(tf.keras.Model):
    '''
    Preprocess for input of pre-training.
    Time series decomposition is applied.
    The components are normalized.
    '''
    def __init__(self,
                 pool_size_trend,
                 nr_of_covariates,
                 sigma,
                 scale_data,
                 **kwargs):
        super().__init__(**kwargs)
        '''
        args:
            pool_size_trend (int): average pool size for trend component.
            nr_of_covariates (int):  number of covariates.
            sigma (float): standard deviation to calculate residuals.
            scale_data (bool): to scale dataset or not.
        '''

        self.scale_data = scale_data

        self.trend_seasonality_tokenizer = TrendSeasonalityTokenizer(
            pool_size_trend=pool_size_trend,
            nr_of_covariates=nr_of_covariates,
            sigma=sigma)

        self.lb_fc_concatter = tf.keras.layers.Concatenate(axis=1)

        if scale_data is True:
            self.data_normalizer = tf.keras.layers.Normalization(axis=None)
            self.data_denormalizer = \
                tf.keras.layers.Normalization(axis=None, invert=True)
        else:
            self.data_normalizer = tf.keras.layers.Identity(trainable=False)
            self.data_denormalizer = tf.keras.layers.Identity(trainable=False)

        self.ts_normalizer = tf.keras.layers.Normalization(axis=1)
        self.ts_denormalizer = \
            tf.keras.layers.Normalization(axis=1, invert=True)

    def adapt(self, inputs):
        '''
        Adapts the mean and standard deviation of components.
        args:
            inputs (tuple) - tuple of 3 elements.
                Each element is a tf.data.Dataset object.
                x_lb: (None, timesteps, covariates)
                x_fc: (None, timesteps, covariates)
                x_ts: (None, features)
        '''
        x_lb, x_fc, x_ts = inputs

        x_lb_fc = self.lb_fc_concatter((x_lb, x_fc))

        if self.scale_data is True:
            self.data_normalizer.adapt(x_lb_fc)
            self.data_denormalizer.adapt(x_lb_fc)

        x_ts = tf.cast(x_ts, dtype=tf.float32)
        self.ts_normalizer.adapt(x_ts)
        self.ts_denormalizer.adapt(x_ts)

    def call(self, inputs, training=False):
        '''
        args:
            inputs (tuple): tuple of 3 elements.
                Each element is a tf.data.Dataset object.
                x_lb: (None, timesteps, covariates)
                x_fc: (None, timesteps, covariates)
                x_ts: (None, features)

        returns:
            y_tre: (None, timesteps, covariates)
            y_sea: (None, timesteps, covariates)
            y_res: (None, timesteps, covariates)
            y_ts: (None, features)
        '''

        x_lb, x_fc, x_ts = inputs

        x_lb_fc = self.lb_fc_concatter((x_lb, x_fc))

        x_lb_fc = self.data_normalizer(x_lb_fc)

        y_tre, y_sea, y_res = self.trend_seasonality_tokenizer(x_lb_fc)

        x_ts = tf.cast(x_ts, dtype=tf.float32)
        y_ts = self.ts_normalizer(x_ts)

        return (y_tre, y_sea, y_res, y_ts)


class InputPreProcessorFT(tf.keras.Model):
    '''
    Preprocess for input of fine-tuning models.
    Trend, seasonility and residual components are prepared.
    The components are normalized if scale_data is True.
    '''
    def __init__(self,
                 pool_size_trend,
                 nr_of_covariates,
                 sigma,
                 scale_data,
                 **kwargs):
        super().__init__(**kwargs)
        '''
        pool_size_trend (int): average pool size for trend component.
        nr_of_covariates (int):  number of covariates.
        sigma (float): standard deviation to calculate residuals.
        scale_data (bool): to scale dataset or not.
        '''

        self.scale_data = scale_data

        self.trend_seasonality_tokenizer = TrendSeasonalityTokenizer(
            pool_size_trend=pool_size_trend,
            nr_of_covariates=nr_of_covariates,
            sigma=sigma)

        if scale_data is True:
            self.data_normalizer = tf.keras.layers.Normalization(axis=None)
            self.data_denormalizer = \
                tf.keras.layers.Normalization(axis=None, invert=True)
        else:
            self.data_normalizer = tf.keras.layers.Identity(trainable=False)
            self.data_denormalizer = tf.keras.layers.Identity(trainable=False)

        self.ts_normalizer = tf.keras.layers.Normalization(axis=1)
        self.ts_denormalizer = tf.keras.layers.Normalization(
            axis=1,
            invert=True)

    def adapt(self, inputs):
        '''
        Adapts the mean and standard deviaiton of
        time series and timestamp features.
        args:
            tuple of 2 elements. Each element is a tf.data.Dataset object.
            x_lb: (None, timesteps, covariates)
            x_ts: (None, features)
        '''
        (x_lb, x_ts) = inputs

        x_ts = tf.cast(x_ts, dtype=tf.float32)

        if self.scale_data is True:
            self.data_normalizer.adapt(x_lb)
            self.data_denormalizer.adapt(x_lb)

        self.ts_normalizer.adapt(x_ts)
        self.ts_denormalizer.adapt(x_ts)

    def call(self, inputs):
        '''
        inputs: tuple of 2 elements.
        Each element is a tf.data.Dataset object.
            x_lb: (None, timesteps, covariates)
            x_ts: (None, features) timestamp features

        returns tuple of 4 elemements.
            y_lb_tre: (None, timesteps, covariates)
            y_lb_sea: (None, timesteps, covariates)
            y_lb_res: (None, timesteps, covariates)
            y_lb_ts: (None, features)
        '''
        (x_lb, x_ts) = inputs

        x_ts = tf.cast(x_ts, dtype=tf.float32)

        x_lb = self.data_normalizer(x_lb)

        y_lb_tre, y_lb_sea, y_lb_res = self.trend_seasonality_tokenizer(x_lb)

        y_ts = self.ts_normalizer(x_ts)

        return (y_lb_tre, y_lb_sea, y_lb_res, y_ts)


class TargetPreProcessor(tf.keras.Model):
    '''Preprocess to produce target features.'''
    def __init__(self, begin_scalar, end_scalar, **kwargs):
        super().__init__(**kwargs)
        self.begin_scalar = begin_scalar
        self.end_scalar = end_scalar

    def call(self, inputs):
        '''
        args:
            x_fc: (None, timesteps, covariates)
        returns:
            y: (None, timesteps, covariates)
            y_shifted: (None, timesteps, covariates)
        '''
        x_fc = inputs

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
