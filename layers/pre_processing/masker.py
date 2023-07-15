
import tensorflow as tf

class PatchMasker(tf.keras.layers.Layer):

    def __init__(self, fMaskingRate, fMskScalar, **kwargs):
        super().__init__(**kwargs)

        self.fMaskingRate = fMaskingRate
        self.fMskScalar = fMskScalar


    
    def call(self, x):
        '''
        inputs: single channel tokenized aspect. (None, nr_of_patches, feature_size)

        maskes some patches randomly.
        
        outputs: masked tokenized aspect. (None, nr_of_patches, feature_size)
        '''
        
        iNrOfSamples=  x.shape[0]
        iNrOfPatches = x.shape[1]
        iNrOfPatchesToMsk = int(self.fMaskingRate * iNrOfPatches)

        aPatchesToMask = tf.random.uniform([iNrOfPatches])

        aPatchesToMask = tf.argsort( aPatchesToMask)[:iNrOfPatchesToMsk]
        aPatchesToMask = tf.sort(aPatchesToMask, axis= 0)

        
        y = tf.add(tf.zeros_like(x),  self.fMskScalar) 

        z = []
        for i in range(iNrOfPatches):
            
            r = tf.constant([])
            if i in aPatchesToMask:
                r = y[:, i]
            else:
                r = x[:, i]


            z.append(r)
                

        z= tf.stack(z, axis =1)

        return z
