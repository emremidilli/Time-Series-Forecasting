import tensorflow as tf
import tensorflow_probability as tfp


class PatchTokenizer(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.trainable = False
        self.reshaper = tf.keras.layers.Reshape((-1, self.patch_size))

    def call(self, x):
        '''
        x: (None, nr_of_time_steps)

        outputs: (None, nr_of_patches, patch_size)
        '''

        y = self.reshaper(x)

        return y


class DistributionTokenizer(tf.keras.layers.Layer):
    def __init__(self, nr_of_bins, fMin, fMax, **kwargs):
        super().__init__(**kwargs)

        self.trainable = False

        self.nr_of_bins = nr_of_bins

        self.bin_boundaries = tf.linspace(
            start=fMin,
            stop=fMax,
            num=self.nr_of_bins - 1
        )

        self.oDiscritizer = tf.keras.layers.Discretization(
            bin_boundaries=self.bin_boundaries)

    def call(self, x):
        '''
        inputs: lookback normalized input (None, timesteps, feature)

        returns: (None, timesteps, feature)
        '''
        y = self.oDiscritizer(x)

        output_list = []
        for i in range(0, self.nr_of_bins):
            output_list.append(tf.math.count_nonzero(y == i, axis=2))

        z = tf.stack(output_list, axis=2)

        z = tf.math.divide(z, tf.expand_dims(tf.reduce_sum(z, axis=2), 2))

        return z


class TrendSeasonalityTokenizer(tf.keras.layers.Layer):
    def __init__(self, pool_size_reduction, pool_size_trend, **kwargs):
        super().__init__(**kwargs)

        self.oAvgPoolReducer = tf.keras.layers.AveragePooling1D(
            pool_size=pool_size_reduction,
            strides=pool_size_reduction,
            padding='valid',
            data_format='channels_first')

        self.oAvgPoolTrend = tf.keras.layers.AveragePooling1D(
            pool_size=pool_size_trend,
            strides=1,
            padding='same',
            data_format='channels_first')

        self.trainable = False

    def call(self, x):
        '''
            x: lookback normalized series that is patched
            (None, nr_of_patches, patch_size)

            for each patch
                reduces the input by avg pooling
                calculate the trend component by avg pooling
                calculate thte seasonality componenet by
                    subtracting the trend componenet from sampled

            returns:  tuple of 2 elements
                1. trend component - (None, nr_of_patches, patch_size)
                2. seasonality component - (None, nr_of_patches, patch_size)
        '''
        x_reduced = self.oAvgPoolReducer(x)

        y_trend = self.oAvgPoolTrend(x_reduced)

        y_seasonality = tf.subtract(x_reduced, y_trend)

        return (y_trend, y_seasonality)


class QuantileTokenizer(tf.keras.layers.Layer):
    def __init__(self, quantiles, **kwargs):
        super().__init__(**kwargs)

        self.trainable = False

        self.percentiles = tf.convert_to_tensor(quantiles) * 100

    def call(self, x):
        '''
        inputs: lookback normalized & patch tokenizer input
            (None, timesteps, feature)
            timesteps are patches.
            features are single steps within the same patch.

        returns: (None, timesteps, feature)
            timesteps are patches.
            features are quantiles.
        '''
        y = tfp.stats.percentile(x, self.percentiles, axis=2)
        y = tf.transpose(y, perm=[1, 2, 0])
        return y
