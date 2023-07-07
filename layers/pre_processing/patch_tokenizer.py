import tensorflow as tf

import sys
sys.path.append( '/home/yunusemre/Time-Series-Forecasting/')
from preprocessing.constants import *



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
    


class LookbackNormalizer(tf.keras.layers.Layer):
    '''
        relaxiation_rate: represents how much the forecast horizon boundaries will exceed the lookback boundaries.
            by default 0. All the values that fall beyond the minimum and maximum of lookback window, will be set 0 and 1 respectively.
    '''

    def __init__(self, relaxiation_rate = 0, **kwargs):
        super().__init__(**kwargs)
        self.relaxiation_rate = relaxiation_rate


    '''
        inputs: tuple of 2 elements.
            1. original lookback series (None, nr_of_lookback_time_steps)
            2. series to normalize (None, nr_of_time_steps)

        outputs: the normalized series (None, nr_of_time_steps)
    '''
    def call(self, inputs):

        x_lb, x = inputs


        

        










        
        



class DistributionTokenizer(tf.keras.layers.Layer):
    
    def __init__(self, num_bins, **kwargs):
        super().__init__(**kwargs)
        
        self.num_bins = num_bins
        
    '''
        inputs: tuple of 3 elements:
            1. x - original input (None, nr_of_patches, patch_size)
            2. fMin - minimum boundary of bins
            3. fMax - maximum boundary of bins
        
        returns: (None, nr_of_patches, num_bins)
    '''    
    def call(self, inputs):
        
        x, fMin, fMax = inputs
        
        bin_boundaries=tf.linspace(
            start = fMin, 
            stop = fMax, 
            num = self.num_bins
        )
        
        oDiscritizer = tf.keras.layers.Discretization(bin_boundaries = bin_boundaries)
        
        y = oDiscritizer(x)
        
        return y
    
    
    
            
            
            
if __name__ == '__main__':
    
    import os
    import pandas as pd
    
    import numpy as np
    

    aFileNames = os.listdir(RAW_DATA_FOLDER)
    
    
    aSampleIds = [3500, 6500]
    aLookback = np.zeros((len(aSampleIds),FORECAST_HORIZON * LOOKBACK_COEFFICIENT, len(aFileNames) ))
    
    
    
    
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
            df = dfRaw.iloc[k-(FORECAST_HORIZON * LOOKBACK_COEFFICIENT):k]
            
            arr = df.loc[:, 'TICKER'].to_numpy()
            

            aLookback[j, :, i] = arr
                        

        
    
    oLookbackTokenizer = PatchTokenizer(PATCH_SIZE)
    oDistTokenizer = DistributionTokenizer(NR_OF_BINS)
    
    x = aLookback[:,:,0].copy()
    x = oLookbackTokenizer(x)
    
    x = oDistTokenizer(x)
    
    
    print(x.shape)
    
    
    
    
    
        