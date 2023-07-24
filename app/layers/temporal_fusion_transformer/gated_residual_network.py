import tensorflow as tf
from layers.temporal_fusion_transformer.gated_linear_unit import gated_linear_unit


class gated_residual_network(tf.keras.layers.Layer):
    '''
      Args:
          iInputDims (int): Size of the input.
          iOutputDims (int): Size of the output.
          fDropout (float): Fraction between 0 and 1 corresponding to the degree of fDropout used.
          bIsWithStaticCovariate (boolean): Size of the static context vector. 
    '''
    def __init__(self, iInputDims ,iOutputDims, fDropout, bIsWithStaticCovariate=False):
        super().__init__()
        
        
        self.iInputDims = iInputDims
        self.iOutputDims = iOutputDims
        self.fDropout = fDropout
        self.bIsWithStaticCovariate = bIsWithStaticCovariate
        
        if self.iInputDims != self.iOutputDims:
            self.oDenseSkipConnection = tf.keras.layers.Dense(units = self.iOutputDims, activation = None) 
        
        # Context vector c
        if self.bIsWithStaticCovariate == True:
            self.oDense_c = tf.keras.layers.Dense(units = self.iOutputDims, activation = None) 

        # Dense & ELU
        self.oDense1 = tf.keras.layers.Dense(units = self.iOutputDims, activation = None)
        self.oElu = tf.keras.layers.ELU()

        # Dense & Dropout
        self.oDense2 = tf.keras.layers.Dense(self.iOutputDims)
        self.oDropout = tf.keras.layers.Dropout(self.fDropout)

        # Gate, Add & Norm
        self.oGate = gated_linear_unit(self.iOutputDims)
        self.oLayerNorm = tf.keras.layers.LayerNormalization()


    '''
        Args:
            inputs (list): contain max 2 parts in order to be able to use as part of TimeDistributed layer. 
            Otherwise TimeDistributed layer don't allow multiple input arguments. 
            The only way to pass multiple arguments to TimeDistributed layer, is to pass as a list:
            should be a list with 2 arguments:
                1.st argument:
                    should a transformation of a time patch in following format (None, 1, model_dims)
                    output of an encoder representation (e.g. DisERT) will be in format of (None, number_of_patches, model_dims)
                    for a time step, there will be for 4 representations coming from DisERT, TicERT, TreERT and SeaERT.
                    in TFT paper, model_dims is mentioned as "transformed inputs" in Figure 2.
                2nd argument: 
                    is optional input. In case static covariates will be an input, it should be accepted.
    '''
    def call(self, inputs):
        x = inputs[0]
        if self.bIsWithStaticCovariate == True:
            c = inputs[1]
        else:
            c = None
         
        a = x # a is used for skip connection
        if self.iInputDims != self.iOutputDims:
            a = self.oDenseSkipConnection(a)
        
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