'''
    Trains a pre-training model for a univariate forecasting model.
    loads training datasets
    builds a pre-training model
    trains and saves pre-training logs

    inputs: 
        lb_train: (None, timesteps)
        fc_train: (None, timesteps)
'''


import sys
sys.path.append( './')

from settings import *

from models import *

import numpy as np


from keras.optimizers import Adam

from keras.losses import  MeanSquaredError
from keras.metrics import  MeanAbsoluteError



if __name__ == '__main__':
    sChannel = 'EURUSD'

    lb_train = np.load(f'{TRAINING_DATA_FOLDER}/{sChannel}/lb_train.npy')[: 256]
    fc_train = np.load(f'{TRAINING_DATA_FOLDER}/{sChannel}/fc_train.npy')[: 256]
    
    oPreProcessor = PreProcessor(
        iPatchSize = PATCH_SIZE,
        fPatchSampleRate = PATCH_SAMPLE_RATE,
        iNrOfBins = NR_OF_BINS
    )

    lb, lb_dist, lb_tre, lb_sea, fc, fc_dist, fc_tre, fc_sea = oPreProcessor((lb_train,fc_train))
    dist = oPreProcessor.concat_lb_fc((lb_dist, fc_dist))
    tre = oPreProcessor.concat_lb_fc((lb_tre, fc_tre))
    sea = oPreProcessor.concat_lb_fc((lb_sea, fc_sea))

    
    oModel = PreTraining(
                 iNrOfEncoderBlocks = 3,
                 iNrOfHeads = 5, 
                 fDropoutRate = 0.10, 
                 iEncoderFfnUnits = 64,
                 iEmbeddingDims = 64,
                 iPatchSize = PATCH_SIZE,
                 fMskRate = MASK_RATE,
                 fMskScalar = MSK_SCALAR,
                 iNrOfBins = NR_OF_BINS,
                 iNrOfPatches= lb.shape[1] + fc.shape[1]
    )


    oModel.compile(
        loss = MeanSquaredError(name = 'mse'),
        metrics = MeanAbsoluteError(name = 'mae'),
        optimizer= Adam(
            learning_rate=1e-4,
            beta_1 = 0.85
        )
    )

    

    oModel.fit(
        (lb_dist,lb_tre, lb_sea,fc_dist, fc_tre , fc_sea), 
        (dist, tre, sea), 
        epochs= NR_OF_EPOCHS, 
        batch_size=MINI_BATCH_SIZE, 
        verbose=1
    )

    ...