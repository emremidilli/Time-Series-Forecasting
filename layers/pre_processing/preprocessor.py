import tensorflow as tf

import sys
sys.path.append( '/home/yunusemre/Time-Series-Forecasting/')
from preprocessing.constants import *



class LookbackNormalizer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    '''
        inputs: tuple of 2 elements.
            1. original lookback series (None, nr_of_lookback_time_steps)
            2. series to normalize (None, nr_of_time_steps)

        outputs: the normalized series (None, nr_of_time_steps)
    '''
    def call(self, inputs):

        x_lb, x = inputs

        aMin = tf.math.reduce_min(x_lb, axis =1)
        aMax = tf.math.reduce_max(x_lb, axis =1)
        y = tf.subtract(aMax, aMin)
        z = tf.subtract(x, tf.expand_dims(aMin, axis = 1))
        r = tf.divide(z, tf.expand_dims(y, axis = 1))

        return r
    
class PatchTokenizer(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        

    '''
        x: (None, nr_of_time_steps)
        
        outputs: (None, nr_of_patches, patch_size)
    '''
    def call(self, x):
        
        y = tf.reshape(x, (x.shape[0], -1, self.patch_size))
        
        return y
    
class DistributionTokenizer(tf.keras.layers.Layer):
    def __init__(self, iNrOfBins, fMin, fMax, **kwargs):
        super().__init__(**kwargs)
        
        self.iNrOfBins = iNrOfBins


        self.bin_boundaries = tf.linspace(
            start = fMin, 
            stop = fMax, 
            num = self.iNrOfBins
        )

        
        self.oDiscritizer = tf.keras.layers.Discretization(bin_boundaries = self.bin_boundaries)

        
    '''
        inputs: lookback normalized input (None, nr_of_patches, patch_size)
        
        returns: (None, nr_of_patches, num_bins)
    '''    
    def call(self, x):
        
        y = self.oDiscritizer(x)

        output_list = []
        for i in range(1, self.iNrOfBins+1):
            output_list.append(tf.math.count_nonzero(y == i, axis = 2) )

        
        z = tf.stack(output_list, axis = 2)

        z = tf.math.divide(z  , tf.expand_dims(tf.reduce_sum(z, axis = 2), 2))

        return z
    
class TrendSeasonalityTokenizer(tf.keras.layers.Layer):
    def __init__(self, iPoolSizeSampling, **kwargs):
        super().__init__(**kwargs)
        

        self.oAvgPool = tf.keras.layers.AveragePooling1D(pool_size = iPoolSizeSampling, strides=1, padding='same', data_format='channels_first')


    '''
        inputs: lookback normalized series that is patched (None, nr_of_patches, patch_size)

        for each patch
            calculate the trend component
            calculate thte seasonality componenet by subtracting the trend componenet from sampled
            

        returns:  tuple of 2 elements
            1. trend component - (None, nr_of_patches, patch_size)
            2. seasonality component - (None, nr_of_patches, patch_size)
    '''
    def call(self, x):

        y_trend = self.oAvgPool(x)

        y_seasonality = tf.subtract(y_trend, x)

        return (y_trend, y_seasonality)
        
class PatchMasker(tf.keras.layers.Layer):

    def __init__(self, fMaskingRate, fMskScalar, **kwargs):
        super().__init__(**kwargs)

        self.fMaskingRate = fMaskingRate
        self.fMskScalar = fMskScalar


    '''
        inputs: single channel tokenized aspect. (None, nr_of_patches, feature_size)

        maskes some patches randomly.
        

        outputs: masked tokenized aspect. (None, nr_of_patches, feature_size)
    '''
    def call(self, x):
        
        
        iNrOfSamples=  x.shape[0]
        iNrOfPatches = x.shape[1]
        iNrOfPatchesToMsk = int(self.fMaskingRate * iNrOfPatches)

        aPatchesToMask = tf.random.uniform([iNrOfPatches])

        aPatchesToMask = tf.argsort( aPatchesToMask)[:iNrOfPatchesToMsk]
        aPatchesToMask = tf.sort(aPatchesToMask, axis= 0)

        
        y = tf.add(tf.zeros_like(x),  self.fMskScalar) 

        z = []
        for i in range(iNrOfPatches):
            
            r = tf.constant([])
            if i in aPatchesToMask:
                r = y[:, i]
            else:
                r = x[:, i]


            z.append(r)
                

        z= tf.stack(z, axis =1)

        return z

class PatchShifter(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



    '''
        inputs: patched input (None, nr_of_patches, feature_size)

        outputs: randomly shifted version (None, nr_of_patches, feature_size)
    '''
    def call(self, x):
        
        iNrOfPatches = x.shape[1]

        i = tf.random.uniform(shape=(), minval=1, maxval=iNrOfPatches-1, dtype=tf.int32)

        y = tf.roll(x, shift = i ,axis = 1)

        return y

