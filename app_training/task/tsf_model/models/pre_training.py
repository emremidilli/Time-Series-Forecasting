import tensorflow as tf

from tsf_model.layers.ConTemPreT import Representation, \
    MppDecoder, ProjectionHead
from tsf_model.layers.pre_processing import PatchMasker, PatchShifter


class PreTraining(tf.keras.Model):
    '''
        Keras model for pre-training purpose.
    '''
    def __init__(self,
                 iNrOfEncoderBlocks,
                 iNrOfHeads,
                 fDropoutRate,
                 iEncoderFfnUnits,
                 embedding_dims,
                 iProjectionHeadUnits,
                 iReducedDims,
                 fMskRate,
                 msk_scalar,
                 iNrOfBins,
                 iNrOfLookbackPatches,
                 iNrOfForecastPatches,
                 **kwargs):
        super().__init__(**kwargs)

        self.margin = 0.10

        self.nr_of_lookback_patches = iNrOfLookbackPatches
        self.nr_of_forecast_patches = iNrOfForecastPatches

        self.patch_masker = PatchMasker(
            masking_rate=fMskRate, msk_scalar=msk_scalar)

        self.patch_shifter = PatchShifter()

        self.encoder_representation = Representation(
            iNrOfEncoderBlocks,
            iNrOfHeads,
            fDropoutRate,
            iEncoderFfnUnits,
            embedding_dims
        )

        self.lookback_forecast_concatter = tf.keras.layers.Concatenate(axis=1)

        self.decoder_dist = MppDecoder(
            iNrOfBins,
            iNrOfLookbackPatches + iNrOfForecastPatches,
            name='decoder_dist')
        self.decoder_tre = MppDecoder(
            iReducedDims,
            iNrOfLookbackPatches + iNrOfForecastPatches,
            name='decoder_tre')
        self.decoder_sea = MppDecoder(
            iReducedDims,
            iNrOfLookbackPatches + iNrOfForecastPatches,
            name='decoder_sea')

        self.projection_head = ProjectionHead(iProjectionHeadUnits,
                                              name='projection_head')

        # learning rate tracker
        self.lr_tracker = tf.keras.metrics.Mean(name='lr')
        # losses
        self.loss_tracker_mpp = tf.keras.metrics.Mean(name='loss_mpp')
        self.loss_tracker_cl = tf.keras.metrics.Mean(name='loss_cl')

        # metrics
        self.mae_dist = tf.keras.metrics.MeanAbsoluteError(name='mae_dist')
        self.mae_tre = tf.keras.metrics.MeanAbsoluteError(name='mae_tre')
        self.mae_sea = tf.keras.metrics.MeanAbsoluteError(name='mae_sea')
        self.cos_dist = tf.keras.metrics.CosineSimilarity(name='cos_dist')
        self.cos_tre = tf.keras.metrics.CosineSimilarity(name='cos_tre')
        self.cos_sea = tf.keras.metrics.CosineSimilarity(name='cos_sea')
        self.cos_true = tf.keras.metrics.CosineSimilarity(name='cos_true')
        self.cos_false = tf.keras.metrics.CosineSimilarity(name='cos_false')

    def compile(self, mae_optimizer, cl_optimizer, **kwargs):
        super().compile(**kwargs)

        self.mae_optimizer = mae_optimizer
        self.cl_optimizer = cl_optimizer

    def get_compile_config(self):
        cfg = super().get_compile_config()
        cfg.update({
            'mae_optimizer': self.mae_optimizer,
            'cl_optimizer': self.cl_optimizer
        })

        return cfg

    def compile_from_config(self, config):
        mae_optimizer = config['mae_optimizer']
        cl_optimizer = config['cl_optimizer']

        self.compile(
            mae_optimizer=mae_optimizer,
            cl_optimizer=cl_optimizer)

    def mask_patches(self, data):
        '''
            Masks both lookback and forecast patches.
            Patch steps to mask are determined randomly.
        '''
        x_dist, x_tre, x_sea = data
        x_lb_dist = x_dist[:, :self.nr_of_lookback_patches]
        x_lb_tre = x_tre[:, :self.nr_of_lookback_patches]
        x_lb_sea = x_sea[:, :self.nr_of_lookback_patches]
        x_fc_dist = x_dist[:, self.nr_of_lookback_patches:]
        x_fc_tre = x_tre[:, self.nr_of_lookback_patches:]
        x_fc_sea = x_sea[:, self.nr_of_lookback_patches:]

        x_lb_dist_msk, x_lb_tre_msk, x_lb_sea_msk = self.patch_masker(
            (x_lb_dist, x_lb_tre, x_lb_sea))
        x_fc_dist_msk, x_fc_tre_msk, x_fc_sea_msk = self.patch_masker(
            (x_fc_dist, x_fc_tre, x_fc_sea))

        x_dist_msk = self.lookback_forecast_concatter(
            [x_lb_dist_msk, x_fc_dist_msk])
        x_tre_msk = self.lookback_forecast_concatter(
            [x_lb_tre_msk, x_fc_tre_msk])
        x_sea_msk = self.lookback_forecast_concatter(
            [x_lb_sea_msk, x_fc_sea_msk])

        return (x_dist_msk, x_tre_msk, x_sea_msk)

    def augment_pairs(self, data):
        '''
            Augments an input. In each augmentation, different patches are
                masked & shifted randomly.
            Only lookbacks are masked. Forecast are not masked.
            In each augmentation, forecast patches are shifted (rolled) by
                random amount.

            returns: tuples of 6 elements. Each element contains merged
                lookback and forecast patches.
        '''
        x_dist, x_tre, x_sea = data
        x_lb_dist = x_dist[:, :self.nr_of_lookback_patches]
        x_lb_tre = x_tre[:, :self.nr_of_lookback_patches]
        x_lb_sea = x_sea[:, :self.nr_of_lookback_patches]
        x_fc_dist = x_dist[:, self.nr_of_lookback_patches:]
        x_fc_tre = x_tre[:, self.nr_of_lookback_patches:]
        x_fc_sea = x_sea[:, self.nr_of_lookback_patches:]

        iNrOfForecastPatches = tf.shape(x_fc_dist)[1]

        # mask
        x_lb_dist_msk, x_lb_tre_msk, x_lb_sea_msk = self.patch_masker(
            (x_lb_dist, x_lb_tre, x_lb_sea))
        x_dist_true = self.lookback_forecast_concatter(
            [x_lb_dist_msk, x_fc_dist])
        x_tre_true = self.lookback_forecast_concatter(
            [x_lb_tre_msk, x_fc_tre])
        x_sea_true = self.lookback_forecast_concatter(
            [x_lb_sea_msk, x_fc_sea])

        # shift
        i = tf.random.uniform(
            shape=[],
            minval=1,
            maxval=iNrOfForecastPatches,
            dtype=tf.int32)
        x_fc_dist_sft, x_fc_tre_sft, x_fc_sea_sft = self.patch_shifter(
            (x_fc_dist, x_fc_tre, x_fc_sea, i))
        x_dist_false = self.lookback_forecast_concatter(
            [x_lb_dist_msk, x_fc_dist_sft])
        x_tre_false = self.lookback_forecast_concatter(
            [x_lb_tre_msk, x_fc_tre_sft])
        x_sea_false = self.lookback_forecast_concatter(
            [x_lb_sea_msk, x_fc_sea_sft])

        return (x_dist_true,
                x_tre_true,
                x_sea_true,
                x_dist_false,
                x_tre_false,
                x_sea_false)

    # @tf.function()
    def train_step(self, data):
        '''
            trains a step in two phases:
                1. masked patch prediction
                2. contrastive learning
        '''
        anchor_dist, anchor_tre, anchor_sea, dates = data

        # masked auto-encoder
        msk_dist, msk_tre, msk_sea = self.mask_patches(
            (anchor_dist, anchor_tre, anchor_sea))

        with tf.GradientTape() as tape:
            y_pred_dist, y_pred_tre, y_pred_sea = self((msk_dist,
                                                        msk_tre,
                                                        msk_sea,
                                                        dates))

            # compute the loss value
            loss_dist = tf.keras.losses.mean_squared_error(
                y_pred=y_pred_dist, y_true=anchor_dist)
            loss_tre = tf.keras.losses.mean_squared_error(
                y_pred=y_pred_tre, y_true=anchor_tre)
            loss_sea = tf.keras.losses.mean_squared_error(
                y_pred=y_pred_sea, y_true=anchor_sea)

            loss_mpp = loss_dist + loss_tre + loss_sea

        # compute gradients
        trainable_vars = self.encoder_representation.trainable_variables + \
            self.decoder_dist.trainable_variables + \
            self.decoder_tre.trainable_variables + \
            self.decoder_sea.trainable_variables
        gradients = tape.gradient(loss_mpp, trainable_vars)

        # update weights
        self.mae_optimizer.apply_gradients(
            zip(gradients, trainable_vars))

        # compute own metrics
        self.loss_tracker_mpp.update_state(loss_mpp)
        self.mae_dist.update_state(y_pred=y_pred_dist, y_true=anchor_dist)
        self.mae_tre.update_state(y_pred=y_pred_tre, y_true=anchor_tre)
        self.mae_sea.update_state(y_pred=y_pred_sea, y_true=anchor_sea)
        self.cos_dist.update_state(y_pred=y_pred_dist, y_true=anchor_dist)
        self.cos_tre.update_state(y_pred=y_pred_tre, y_true=anchor_tre)
        self.cos_sea.update_state(y_pred=y_pred_sea, y_true=anchor_sea)

        # contrastive learning
        dist_true, tre_true, sea_true, dist_false, tre_false, sea_false = \
            self.augment_pairs((anchor_dist, anchor_tre, anchor_sea))

        with tf.GradientTape() as tape:
            x_cont_temp_true = self.encoder_representation(
                (dist_true, tre_true, sea_true, dates))
            x_cont_temp_false = self.encoder_representation(
                (dist_false, tre_false, sea_false, dates))
            x_cont_temp_anchor = self.encoder_representation(
                (anchor_dist, anchor_tre, anchor_sea, dates))

            y_logits_false = self.projection_head(x_cont_temp_false)
            y_logits_true = self.projection_head(x_cont_temp_true)
            y_logits_anchor = self.projection_head(x_cont_temp_anchor)

            # compute the loss value
            distance_true = tf.reduce_sum(
                tf.square(y_logits_anchor - y_logits_true), -1)
            distance_false = tf.reduce_sum(
                tf.square(y_logits_anchor - y_logits_false), -1)
            loss_cl = tf.maximum(
                distance_true - distance_false + self.margin, 0.0)

        # compute gradients
        trainable_vars = self.encoder_representation.trainable_variables + \
            self.projection_head.trainable_variables
        gradients = tape.gradient(loss_cl, trainable_vars)

        # update weights
        self.cl_optimizer.apply_gradients(
            zip(gradients, trainable_vars))

        # compute own metrics
        self.loss_tracker_cl.update_state(loss_cl)
        self.cos_true.update_state(
            y_true=y_logits_anchor, y_pred=y_logits_true)
        self.cos_false.update_state(
            y_true=y_logits_anchor, y_pred=y_logits_false)

        self.lr_tracker.update_state(self.cl_optimizer.lr)

        dic = {
            'loss_mpp': self.loss_tracker_mpp.result(),
            'loss_cl': self.loss_tracker_cl.result(),
            'mae_dist': self.mae_dist.result(),
            'mae_tre': self.mae_tre.result(),
            'mae_sea': self.mae_sea.result(),
            'cos_dist': self.cos_dist.result(),
            'cos_tre': self.cos_tre.result(),
            'cos_sea': self.cos_sea.result(),
            'cos_true': self.cos_true.result(),
            'cos_false': self.cos_false.result(),
            'lr': self.lr_tracker.result()
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
            self.loss_tracker_cl,
            self.lr_tracker,
            self.mae_dist,
            self.mae_tre,
            self.mae_sea,
            self.cos_dist,
            self.cos_tre,
            self.cos_sea,
            self.cos_true,
            self.cos_false]

    def call(self, inputs):
        '''
            input: tuple of 4 arrays.
                1. dist: (none, timesteps, features)
                2. tre: (none, timesteps, features)
                3. sea: (none, timesteps, features)
                4. date: (none, features)
        '''
        y_cont_temp = self.encoder_representation(inputs)

        y_pred_dist = self.decoder_dist(y_cont_temp)
        y_pred_tre = self.decoder_tre(y_cont_temp)
        y_pred_sea = self.decoder_sea(y_cont_temp)

        return (y_pred_dist, y_pred_tre, y_pred_sea)
