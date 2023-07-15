'''
    Trains a pre-training model for a single channel.

    Inputs:
        lb_train, fc_train datasets.
'''


import sys
sys.path.append( './')

from settings import *

from models.pre_training import *

import numpy as np


if __name__ == '__main__':
    '''
        loads training datasets
        builds a pre-training model
        trains and saves pre-training logs
    '''


    sChannel = 'EURUSD'

    lb_train = np.load(f'{TRAINING_DATA_FOLDER}/{sChannel}/lb_train.npy')[: 256]
    fc_train = np.load(f'{TRAINING_DATA_FOLDER}/{sChannel}/fc_train.npy')[: 256]

    
    
    oModel = PreTraining(
                 iNrOfEncoderBlocks = 3,
                 iNrOfHeads = 5, 
                 fDropoutRate = 0.10, 
                 iEncoderFfnUnits = 64,
                 iEmbeddingDims = 64,
                 iPatchSize = PATCH_SIZE,
                 fPatchSampleRate = PATCH_SAMPLE_RATE,
                 fMskRate = MASK_RATE,
                 fMskScalar = MSK_SCALAR,
                 iNrOfBins = NR_OF_BINS
    )


    y = oModel((lb_train,fc_train ))



    ...
    