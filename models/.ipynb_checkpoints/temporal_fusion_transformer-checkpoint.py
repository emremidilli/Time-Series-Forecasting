import sys
sys.path.append( '../')
from layers.variable_selection_network import variable_selection_network

import tensorflow as tf

class temporal_fusion_transformer(tf.keras.Model):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
        self. oVsnLookback = variable_selection_network(
            iModelDims = 32,
            iNrOfChannels = 3 ,
            fDropout = 0.1
        )
        self.oTimeDistVsnLookback = tf.keras.layers.TimeDistributed(self.oVsnLookback)
        
        
        self.oVsnForecast = variable_selection_network(
            iModelDims = 32,
            iNrOfChannels = 3 ,
            fDropout = 0.1
        )
        self.oTimeDistVsnForecast = tf.keras.layers.TimeDistributed(self.oVsnForecast)
        
        
        self.oLstmEncoder = tf.keras.layers.LSTM(
            units = 32,
            return_sequences=True, 
            return_state=True
        )
        
        self.oLstmDecoder = tf.keras.layers.LSTM(
            units = 32,
            return_sequences=True, 
            return_state=True
        )
        
        
        
    def call(self, x):
        x_l = x[0]
        c_s_l = x[1]
        x_f = x[2]
        c_s_f = x[3]
        
        y_l, v_l = self.oTimeDistVsnLookback([x_l, c_s_l])
        y_encoder, h_encoder, c_encoder = self.oLstmEncoder(y_l)
    
        y_f, v_f = self.oTimeDistVsnForecast([x_f, c_s_f])
        y_decoder, h_decoder, c_decoder = self.oLstmDecoder(y_f)
        
        return y_decoder
        
        