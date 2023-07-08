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
    
class DigitNormalizer(tf.keras.layers.Layer):
    def __init__(self, iStaticDigits, **kwargs):
        super().__init__(**kwargs)
        self.iStaticDigits = iStaticDigits
        self.trainable = False
    
    '''
        inputs: tuple of 2 elements.
            1. original lookback series (None, nr_of_lookback_time_steps)
            2. series to normalize (None, nr_of_time_steps)

        outputs: tuple of 3 elements.
            1. the normalized series (None, nr_of_time_steps)
            2. static digits of series (None, 1)
            3. number of transitions within the series (None, 1)
    '''
    def call(self, inputs):

        x_lb, x = inputs

        multiplier = tf.constant(10**self.iStaticDigits, dtype=x.dtype)

        x_static = tf.round(x * multiplier) / multiplier

        x_dynamic = tf.subtract(x, x_static)

        x_nr_of_transitions = tf.subtract(tf.math.reduce_max(x_static, axis = 1) , tf.math.reduce_min(x_static, axis = 1))
        x_nr_of_transitions = tf.divide( x_nr_of_transitions , tf.constant(10**-self.iStaticDigits, dtype=x.dtype))
        x_nr_of_transitions = tf.round(x_nr_of_transitions, 0)

        return (x_dynamic, x_static, x_nr_of_transitions)



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
        return z
    
class TickerTokenizer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



    '''
        inputs: lookback normalized input (None, nr_of_patches, patch_size)

        returns: (None, nr_of_patches, num_bins)
    '''
    def call(self, x):
        ...
    
            
            
            
if __name__ == '__main__':
    
    import os
    import pandas as pd
    
    import numpy as np
    
 

    aFileNames = os.listdir(RAW_DATA_FOLDER)
    
    
    aSampleIds = [3500, 6500]
    aLookback = np.zeros((len(aSampleIds),FORECAST_HORIZON * LOOKBACK_COEFFICIENT, len(aFileNames) ))
    aForecast = np.zeros((len(aSampleIds),FORECAST_HORIZON, len(aFileNames) ))
    
    
    
    
    for i , sFileName in enumerate(aFileNames):
        
        print(f'Converting file {sFileName}')
        
        dfRaw = pd.read_csv(
            f'{RAW_DATA_FOLDER}/{sFileName}', 
            delimiter='\t',
            usecols=['<DATE>', '<TIME>','<HIGH>', '<VOL>']
        )
        
        dfRaw.loc[:, 'TIME_STAMP'] = dfRaw.loc[:, '<DATE>'] + ' ' +dfRaw.loc[:, '<TIME>']
        dfRaw.drop(['<DATE>', '<TIME>'], axis = 1, inplace = True)


        dfRaw.rename(columns = {'<HIGH>':'TICKER',
                             '<VOL>':'OBSERVED'
                            }, inplace = True)
        
        
        
        
        for j, k in enumerate(aSampleIds):

            aLookback[j, :, i] = dfRaw.iloc[k-(FORECAST_HORIZON * LOOKBACK_COEFFICIENT):k].loc[:, 'TICKER'].to_numpy()
            aForecast[j, :, i] = dfRaw.iloc[k:k+FORECAST_HORIZON].loc[:, 'TICKER'].to_numpy()


                        

        
    oDigitNormalizer = DigitNormalizer(2)
    oLookbackNormalizer = LookbackNormalizer()
    oPatchTokenizer = PatchTokenizer(PATCH_SIZE)
    oDistTokenizer = DistributionTokenizer(
        iNrOfBins=NR_OF_BINS,
        fMin=0,
        fMax=1 #relaxation can be applied. (eg. tredinding series)
        )
    
    
    
    x_lb = aLookback[:,:,0].copy()
    x_fc = aForecast[:,:,0].copy()


    x_lb_dynamic, x_lb_static, x_lb_nr_of_transitions = oDigitNormalizer(x_lb)    
    
    """     
    x_fc = oLookbackNormalizer((x_lb,x_fc))
    x_lb = oLookbackNormalizer((x_lb,x_lb))
    
    x_lb = oPatchTokenizer(x_lb)
    x_fc = oPatchTokenizer(x_fc)

    x_lb_dist = oDistTokenizer(x_lb)
    x_fc_dist = oDistTokenizer(x_fc)
    """








    print('successful !')
    
    
    
    
    
        