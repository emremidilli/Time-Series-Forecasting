import sys
sys.path.append( './')

import tensorflow as tf

from layers.pre_processing import *
from layers.general_pre_training import *

class PreTraining(tf.keras.Model):
    '''
        Keras model for pre-training purpose.
    '''
    def __init__(self,
                 iNrOfEncoderBlocks,
                 iNrOfHeads, 
                 fDropoutRate, 
                 iEncoderFfnUnits,
                 iEmbeddingDims,
                 iProjectionHeadUnits,
                 iPatchSize,
                 fMskRate,
                 fMskScalar,
                 iNrOfBins,
                 iNrOfPatches,
                 **kwargs):
        super().__init__(**kwargs)

        self.temperature = 0.10

        self.patch_masker = PatchMasker(fMaskingRate=fMskRate, fMskScalar=fMskScalar)

        self.patch_shifter = PatchShifter()
        
        self.encoder_representation = Representation(
            iNrOfEncoderBlocks,
            iNrOfHeads,
            fDropoutRate, 
            iEncoderFfnUnits,
            iEmbeddingDims
        )


        self.lookback_forecast_concatter = tf.keras.layers.Concatenate(axis = 1)


        self.decoder_dist = MppDecoder(iNrOfBins, iNrOfPatches)
        self.decoder_tre = MppDecoder(iPatchSize, iNrOfPatches)
        self.decoder_sea = MppDecoder(iPatchSize, iNrOfPatches)


        self.projection_head = ProjectionHead(iProjectionHeadUnits)

    def compile(self, contrastive_optimizer, masked_autoencoder_optimizer , **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.masked_autoencoder_optimizer = masked_autoencoder_optimizer
        # losses
        self.loss_tracker_mpp = tf.keras.metrics.Mean(name='loss_mpp')
        self.loss_tracker_cl = tf.keras.metrics.Mean(name='loss_cl')

        # metrics
        self.mae_dist = tf.keras.metrics.MeanAbsoluteError(name='mae_dist')
        self.mae_tre = tf.keras.metrics.MeanAbsoluteError(name='mae_tre')
        self.mae_sea = tf.keras.metrics.MeanAbsoluteError(name='mae_sea')
        self.acc_cl = tf.keras.metrics.SparseCategoricalAccuracy(name = 'acc_cl')

    @tf.function
    def mask_patches(self,data):
        '''
            Masks both lookback and forecast patches randomly.
        '''
        x_lb_dist,x_lb_tre, x_lb_sea,x_fc_dist, x_fc_tre , x_fc_sea = data
        x_lb_dist_msk, x_lb_tre_msk, x_lb_sea_msk = self.patch_masker((x_lb_dist, x_lb_tre, x_lb_sea))
        x_fc_dist_msk, x_fc_tre_msk, x_fc_sea_msk = self.patch_masker((x_fc_dist, x_fc_tre, x_fc_sea))

        x_dist_msk = self.lookback_forecast_concatter([x_lb_dist_msk, x_fc_dist_msk])
        x_tre_msk = self.lookback_forecast_concatter([x_lb_tre_msk, x_fc_tre_msk])
        x_sea_msk = self.lookback_forecast_concatter([x_lb_sea_msk, x_fc_sea_msk])


        return (x_dist_msk, x_tre_msk, x_sea_msk)

    @tf.function
    def augment_pairs(self,data):
        '''
            Augments an input as many as its number of forecast patches minus 1.
            In each augmentation, different patches are masked randomly.
            Only lookbacks are masked.
            Forecast are not masked.
            In each augmentation, forecast patches are shifted (rolled).

            returns: tuples of 3 elements. Each element contains merged lookback and forecast patches.
        '''
        x_lb_dist,x_lb_tre, x_lb_sea,x_fc_dist, x_fc_tre , x_fc_sea = data
        
        iNrOfForecastPatches = tf.shape(x_fc_dist)[1]
        iBatchSize= tf.shape(x_fc_dist)[0]

        
        dist_aug = tf.TensorArray(tf.float64,  size=0, dynamic_size=True, clear_after_read=False)
        tre_aug = tf.TensorArray(tf.float64,  size=0, dynamic_size=True, clear_after_read=False)
        sea_aug = tf.TensorArray(tf.float64,  size=0, dynamic_size=True, clear_after_read=False)
        labels_aug = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)

        j = 0
        for i in tf.range(iNrOfForecastPatches-1):
            # mask
            x_lb_dist_msk, x_lb_tre_msk, x_lb_sea_msk = self.patch_masker((x_lb_dist, x_lb_tre, x_lb_sea))
            x_dist_msk = self.lookback_forecast_concatter([x_lb_dist_msk, x_fc_dist])
            x_tre_msk = self.lookback_forecast_concatter([x_lb_tre_msk, x_fc_tre])
            x_sea_msk = self.lookback_forecast_concatter([x_lb_sea_msk, x_fc_sea])

            # shift
            x_fc_dist_sft, x_fc_tre_sft, x_fc_sea_sft = self.patch_shifter((x_fc_dist, x_fc_tre, x_fc_sea, i+1)) # does not shift randomly but based on i.
            x_dist_sft = self.lookback_forecast_concatter([x_lb_dist, x_fc_dist_sft])
            x_tre_sft = self.lookback_forecast_concatter([x_lb_tre, x_fc_tre_sft])
            x_sea_sft = self.lookback_forecast_concatter([x_lb_sea, x_fc_sea_sft])

            labels_true = tf.range(iBatchSize)
            labels_false = tf.range(start=iBatchSize, limit=tf.multiply(iBatchSize , 2))


            # cast
            x_dist_msk = tf.cast(x_dist_msk, tf.float64)
            x_tre_msk = tf.cast(x_tre_msk, tf.float64)
            x_sea_msk = tf.cast(x_sea_msk, tf.float64)
            x_dist_sft = tf.cast(x_dist_sft, tf.float64)
            x_tre_sft = tf.cast(x_tre_sft, tf.float64)
            x_sea_sft = tf.cast(x_sea_sft, tf.float64)


            dist_aug = dist_aug.write(j, x_dist_msk)
            tre_aug= tre_aug.write(j, x_tre_msk)
            sea_aug= sea_aug.write(j, x_sea_msk)
            labels_aug = labels_aug.write(j, labels_true)

            
            j = tf.add(j, 1)

            dist_aug = dist_aug.write(j,x_dist_sft)
            tre_aug = tre_aug.write(j,x_tre_sft)
            sea_aug = sea_aug.write(j,x_sea_sft)
            labels_aug = labels_aug.write(j,labels_false)

            j = tf.add(j, 1)
            
        
        dist_aug = dist_aug.concat()
        tre_aug = tre_aug.concat()
        sea_aug = sea_aug.concat()
        labels_aug = labels_aug.concat()



        return (dist_aug, tre_aug, sea_aug,labels_aug)

    @tf.function
    def train_step(self, data):
        '''
            trains a step in two phases:
                1. masked patch prediction
                2. contrastive learning
        '''
        inputs, outputs = data 

        y_true_dist, y_true_tre, y_true_sea = outputs

        
        # masked auto-encoder
        x_msk = self.mask_patches(inputs)
        with tf.GradientTape() as tape:
            y_cont_temp = self(x_msk, training=True) 
            y_pred_dist = self.decoder_dist(y_cont_temp)
            y_pred_tre = self.decoder_tre(y_cont_temp)
            y_pred_sea = self.decoder_sea(y_cont_temp)


            # compute the loss value
            loss_dist = tf.keras.losses.mean_squared_error(y_pred=y_pred_dist, y_true=y_true_dist)
            loss_tre = tf.keras.losses.mean_squared_error(y_pred=y_pred_tre, y_true=y_true_tre)
            loss_sea = tf.keras.losses.mean_squared_error(y_pred=y_pred_sea, y_true=y_true_sea)

            loss_mpp = loss_dist + loss_tre + loss_sea
            

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_mpp, trainable_vars)

        # update weights
        self.masked_autoencoder_optimizer.apply_gradients(zip(gradients, trainable_vars))

        # compute own metrics
        self.loss_tracker_mpp.update_state(loss_mpp)
        self.mae_dist.update_state(y_pred= y_pred_dist ,y_true= y_true_dist)
        self.mae_tre.update_state(y_pred= y_pred_tre ,y_true= y_true_tre)
        self.mae_sea.update_state(y_pred= y_pred_sea ,y_true= y_true_sea)
        
        

        # contrastive learning
        dist_aug, tre_aug, sea_aug,labels_aug = self.augment_pairs(inputs)
        with tf.GradientTape() as tape:
            x_cont_temp =self((dist_aug, tre_aug, sea_aug), training= True)
            y_logits = self.projection_head(x_cont_temp)

        
            # compute the loss value
            y_logits = tf.math.l2_normalize(y_logits, axis = 1)
            similarities  = (
                tf.matmul(y_logits, y_logits, transpose_b=True)/self.temperature
            )
            similarities = tf.math.sigmoid(similarities)
            similarities = tf.nn.softmax(similarities , axis = 1)

            loss_cl = tf.keras.losses.sparse_categorical_crossentropy(y_pred=similarities, y_true = labels_aug, from_logits = False)

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_cl, trainable_vars)
        
        # update weights
        self.contrastive_optimizer.apply_gradients(zip(gradients, trainable_vars))

        # compute own metrics
        self.loss_tracker_cl.update_state(loss_cl)
        self.acc_cl.update_state(y_pred= similarities ,y_true= labels_aug)


        dic = {
            'loss_mpp': self.loss_tracker_mpp.result(),
            'loss_cl': self.loss_tracker_cl.result(),
            'mae_dist': self.mae_dist.result(),
            'mae_tre': self.mae_tre.result(),
            'mae_sea': self.mae_sea.result(),
            'acc_cl': self.acc_cl.result()
        }

        return dic
    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_tracker_mpp, 
            self.loss_tracker_cl ,
            self.mae_dist, 
            self.mae_tre,
            self.mae_sea,
            self.acc_cl
            ]

    def call(self, inputs, training = False):
        '''
            Inputs: tuple of elements where each of them has format of (None, timesteps, feature).
        '''
        if training == True:
            x_dist, x_tre, x_sea  = inputs
        else:
            x_lb_dist,x_lb_tre, x_lb_sea,x_fc_dist, x_fc_tre , x_fc_sea = inputs
            x_dist = self.lookback_forecast_concatter([x_lb_dist, x_fc_dist])
            x_tre = self.lookback_forecast_concatter([x_lb_tre, x_fc_tre])
            x_sea = self.lookback_forecast_concatter([x_lb_sea, x_fc_sea])
            
        x_cont_temp = self.encoder_representation((x_dist, x_tre, x_sea))

        return x_cont_temp