from layers import TrendSeasonalityTokenizer

import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class InputPreProcessor(tf.keras.Model):
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


@tf.keras.saving.register_keras_serializable()
class TargetPreProcessor(tf.keras.Model):
    '''Preprocess to produce target features.'''
    def __init__(self, scale_data, **kwargs):
        super().__init__(**kwargs)

        self.scale_data = scale_data
        if scale_data is True:
            self.data_normalizer = tf.keras.layers.Normalization(axis=None)
            self.data_denormalizer = \
                tf.keras.layers.Normalization(axis=None, invert=True)
        else:
            self.data_normalizer = tf.keras.layers.Identity(trainable=False)
            self.data_denormalizer = tf.keras.layers.Identity(trainable=False)

    def adapt(self, inputs):
        '''
        Adapts the mean and standard deviaiton of
        time series and timestamp features.
        args:
            tuple of 2 elements. Each element is a tf.data.Dataset object.
            x_lb: (None, timesteps, covariates)
            x_ts: (None, features)
        '''
        x_fc = inputs

        if self.scale_data is True:
            self.data_normalizer.adapt(x_fc)
            self.data_denormalizer.adapt(x_fc)

    def call(self, inputs):
        '''
        args:
            x_fc: (None, timesteps, covariates)
        returns:
            y: (None, timesteps, covariates)
            y_shifted: (None, timesteps, covariates)
        '''
        x_fc = inputs

        y = self.data_normalizer(x_fc)

        return y
