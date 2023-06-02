import tensorflow as tf

from layers.temporal_fusion_transformer.gated_residual_network import gated_residual_network

class variable_selection_network(tf.keras.layers.Layer):
    '''
      Args:
          iModelDims: common model_dims for all input representations
          iNrOfChannels: number of channels. For DisERT, TicERT, TreERT, SeaERT, KnoERT and ObsERT there should be seperate representation.
          fDropout: dropout rate fro GRUs.
    '''
    def __init__(self, 
                 iModelDims,
                 iNrOfVariables,
                 fDropout,
                 bIsWithExternal = False
                ):
        super().__init__()
        
        self.iModelDims = iModelDims
        self.fDropout = fDropout
        self.bIsWithExternal = bIsWithExternal
        
        self.aGrus = []
         
        for i in range(iNrOfVariables):
            self.aGrus.append(
                gated_residual_network(
                    iInputDims = iModelDims,
                    iOutputDims = iModelDims, 
                    fDropout=  fDropout, 
                    bIsWithStaticCovariate=False
                )
            )
            # weights of GRUs are not shared accross each variable.
            # but they are shared accross each time patch.
            

        self.oFlatten = tf.keras.layers.Flatten()
        self.oGruFlatten = gated_residual_network(
            iInputDims = iModelDims * iNrOfVariables, 
            iOutputDims = iNrOfVariables, 
            fDropout=  fDropout, 
            bIsWithStaticCovariate=bIsWithExternal
        )
        self.oSoftmax = tf.keras.layers.Softmax(axis=-1)
        
    '''
        Args:
            x: (None, 1, model_dims, nr_of_variables)
               combination of all encoder representations for a given time patch.
               in TFT paper, 
                   - nr_of_variables: is mentioned as dimension of \mu_{\chi}
                   - model_dims: is mentioned as transformed inputs (or embedding)
            c_s: (None, 1, model_dims x nr_of_channels)
               encoded static covariate vector.
    '''
    def call(self, inputs):
        
        
        if self.bIsWithExternal == True:
            x = inputs[0]
            a = self.oFlatten(x)
            c_s = inputs[1]
            v = self.oGruFlatten(a, c_s)
            
        else:
            x = inputs
            a = self.oFlatten(x) 
            v = self.oGruFlatten(a)
            
        v = self.oSoftmax(v)
        v = tf.expand_dims(v, 1)
    
        
    
        arr = []
        i = 0
        for oGru in self.aGrus:
            arr.append(oGru(x[:,:,i]))
            i = i + 1
            
        b = tf.stack(arr, axis = -1)  
        
        c = b * v
        c = tf.reduce_sum(c, -1)
        
        return c, v