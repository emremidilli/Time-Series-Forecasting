import tensorflow as tf

import sys
sys.path.append( '../')

from layers.bootstrap_aggregation.spp_decoder import Spp_Decoder

from layers.bootstrap_aggregation.rpp_decoder import Rpp_Decoder
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC


class bootstrap_aggregation(tf.keras.Model):
    '''
    Args:
        aEncoderRepresentations (array): array where each element is instance of general_pre_training_model.
    '''
        
    def __init__(self, aEncoderRepresentations, **kwargs):
        super().__init__(**kwargs)
        
        self.iNrOfChannels = 3
        self.iNrOfQuantiles = 3

        self.oLoss = BinaryCrossentropy()
        self.oMetric = AUC(name = 'AUC')
        
        self.aEncoderRepresentations = aEncoderRepresentations
        
        self.oDenseAggregater = tf.keras.layers.Dense(units = 1, use_bias = False, activation = None) # bias is not used for aggregated model not to have impact of bias.
        
        self.oSppDecoder = Spp_Decoder(
            iFfnUnits = self.iNrOfQuantiles * 3 # due to quantiles. 3 means one-hot categories of signs {positive, negative and zero}
        )        

        self.oRppDecoder = Rpp_Decoder(
            iFfnUnits = self.iNrOfQuantiles * (self.iNrOfChannels + 1) # +1 is because 0 rank is assigned for the positions of special tokens.
        )
        
        
    '''
    Args:
        x: (None, nr_of_positions, nr_of_features)
        
    Returns:
        a list
            1st element: spp output
            2nd element: rpp output
    '''    

    def call(self, x):
        
        x_bootstrapped = []
        for oEncoderRepresentation in self.aEncoderRepresentations:
            x_bootstrapped.append(oEncoderRepresentation(x))
            
        # (None, nr_of_positions, model_dims, nr_of_models)
        x_bootstrapped = tf.stack(x_bootstrapped, axis =  3) 
        
        x_aggregated = self.oDenseAggregater(x_bootstrapped)
        
        x_aggregated = tf.squeeze(x_aggregated, 3)
        
        y_spp = self.oSppDecoder(x_aggregated)
        
        y_rpp = self.oRppDecoder(x_aggregated)
        
        return [y_spp, y_rpp]
    

        
