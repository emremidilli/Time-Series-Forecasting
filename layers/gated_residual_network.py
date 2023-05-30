import tensorflow as tf
from gated_linear_unit import gated_linear_unit
class gated_residual_network(tf.keras.layers.Layer):
    '''
      Args:
          iInputSize (int): Size of the input
          iHiddenSize (int): Size of the hidden layer
          iOutputSize (int): Size of the output layer
          fDropout (float): Fraction between 0 and 1 corresponding to the degree of fDropout used
          iContextSize (int): Size of the static context vector
    '''
    def __init__(self, iInputSize, iHiddenSize, iOutputSize, fDropout, iContextSize=None):
        super().__init__()

        self.iInputSize = iInputSize
        self.iOutputSize = iOutputSize
        self.iContextSize = iContextSize
        self.iHiddenSize = iHiddenSize
        self.fDropout = fDropout

        if self.iInputSize != self.iOutputSize:
            self.oSkipConnectionDense = tf.keras.layers.Dense(units = self.iOutputSize, activation = None)

        # Context vector c
        if self.iContextSize != None:
            self.oDense_c = tf.keras.layers.Dense(units = self.iHiddenSize, activation = None) 

        # Dense & ELU
        self.oDense1 = tf.keras.layers.Dense(units = self.iHiddenSize, activation = None)
        self.oElu = tf.keras.layers.ELU()

        # Dense & Dropout
        self.oDense2 = tf.keras.layers.Dense(self.iHiddenSize,  self.iOutputSize)
        self.oDropout = tf.keras.layers.Dropout(self.fDropout)

        # Gate, Add & Norm
        self.oGate = gated_linear_unit(self.iOutputSize)
        self.oLayerNorm = tf.keras.layers.LayerNormalization()

        
    def call(self, x, c=None):
        
        if self.iInputSize!=self.iOutputSize:
            a = self.oSkipConnectionDense(x)
        else:
            a = x
        
        x = self.oDense1(x)

        if c != None:
            c = self.oDense_c(c)
            x = x + c

        x = self.oElu(x)
        
        x = self.oDense2(x)
        x = self.oDropout(x)

        x = self.oGate(x)
        x = x + a
        x = self.oLayerNorm(x)
        
        return x