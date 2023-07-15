'''
    LSTM layer that processes sequential input.
'''
import tensorflow as tf


class Position_Embedding(tf.keras.layers.Layer):
    
    def __init__(self, iUnits,**kwargs):
        
        super().__init__(**kwargs)
        
        self.oLstm = tf.keras.layers.LSTM(units=iUnits, return_sequences=True)
        


    def call(self, x):
        '''
            process the sequential input

            input: (None, timesteps, feature)

            output: (None, timesteps, iUnits)
        '''
        
        y = self.oLstm(x)

        return y
    
    
    

    
    