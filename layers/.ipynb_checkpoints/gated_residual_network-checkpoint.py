import tensorflow as tf
from gated_linear_unit import gated_linear_unit
class gated_residual_network(tf.keras.layers.Layer):
    '''
      Args:
          iModelDims (int): Size of the hidden layer. Normally, input, hidden and output sizes are different. 
                            However, our paper already plans to input transformed inputs from pre-trained encoders.
                            That's why all inputs will have dimension of model_dims.
          fDropout (float): Fraction between 0 and 1 corresponding to the degree of fDropout used
          bIsWithStaticCovariate (boolean): Size of the static context vector. 
    '''
    def __init__(self, iModelDims, fDropout, bIsWithStaticCovariate=False):
        super().__init__()

        self.bIsWithStaticCovariate = bIsWithStaticCovariate
        self.iModelDims = iModelDims
        self.fDropout = fDropout
        
        # Context vector c
        if self.bIsWithStaticCovariate == True:
            self.oDense_c = tf.keras.layers.Dense(units = self.iModelDims, activation = None) 

        # Dense & ELU
        self.oDense1 = tf.keras.layers.Dense(units = self.iModelDims, activation = None)
        self.oElu = tf.keras.layers.ELU()

        # Dense & Dropout
        self.oDense2 = tf.keras.layers.Dense(self.iModelDims)
        self.oDropout = tf.keras.layers.Dropout(self.fDropout)

        # Gate, Add & Norm
        self.oGate = gated_linear_unit(self.iModelDims)
        self.oLayerNorm = tf.keras.layers.LayerNormalization()

        
        
        

    '''
        Args:
            x: should a transformation of a time patch in following format (None, 1, model_dims)
               output of an encoder representation (e.g. DisERT) will be in format of (None, number_of_patches, model_dims)
               for a time step, there will be for 4 representations coming from DisERT, TicERT, TreERT and SeaERT.
               in TFT paper, model_dims is mentioned as "transformed inputs" in Figure 2.
            c: is optional input. In case static covariates will be an input, it should be accepted.
    '''
    def call(self, x, c=None):

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