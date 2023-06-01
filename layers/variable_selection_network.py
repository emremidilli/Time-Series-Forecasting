import tensorflow as tf

from layers.gated_residual_network import gated_residual_network

class variable_selection_network(tf.keras.layers.Layer):
    '''
      Args:
          iModelDims: common model_dims for all input representations
          iNrOfKnownVars: array since there may be multiple known features
          iNrOfObservedVars: array since there may be multiple observed features
          iNrOfChannels: number of channels. For DisERT, TicERT, TreERT, SeaERT there should be seperate representation.
          fDropout: dropout rate fro GRUs.
    '''
    def __init__(self, 
                 iModelDims,
                 iNrOfKnownVars,
                 iNrOfObservedVars,
                 iNrOfChannels,
                 fDropout
                ):
        super().__init__()
        
        self.iModelDims = iModelDims
        self.fDropout = fDropout
        
        self.aGrus = []
        iTotalVars = (4 + iNrOfKnownVars + iNrOfObservedVars) * iNrOfChannels
        for i in range(iTotalVars):
            self.aGrus.append(
                gated_residual_network(iModelDims, fDropout, bIsWithStaticCovariate=False)
            )
            # weights of GRUs are not shared accross each variable.
            # but they are shared accross each time patch.
            

        self.oFlatten = tf.keras.layers.Flatten()
        self.oGruFlatten = gated_residual_network(iTotalVars * iModelDims, fDropout, bIsWithStaticCovariate=True)
        self.oSoftmax = tf.keras.layers.Softmax(dim=-1)
        
    '''
        Args:
            x: (None, 1, model_dims, nr_of_variables)
               combination of all encoder representations for a given time patch.
               nr_of_variables = (4 + number_of_knowns + number_of_observeds) x nr_of_channels
               in TFT paper, 
                   - nr_of_variables: is mentioned as dimension of \mu_{\chi}
                   - model_dims: is mentioned as transformed inputs (or embedding)
            c_s: (None, 1, model_dims x nr_of_channels)
               encoded static covariate vector.
    '''
    def call(self, x, c_s):
        a = self.oFlatten(x)
        v = self.oGruFlatten(a, c_s)
        
        v = self.oSoftmax(v)
        
        arr = []
        i = 0
        for oGru in self.aGrus:
            arr.append(oGru(x[:,:,i]))
            i = i + 1
            
        b = tf.stack(arr)    
        
        c = b * v
        c = tf.reduce_sum(c, -1)
        
        return c