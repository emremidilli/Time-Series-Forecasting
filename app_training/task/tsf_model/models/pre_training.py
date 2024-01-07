import tensorflow as tf

from tsf_model.layers import Representation, \
    MppDecoder, ProjectionHead, PatchMasker, PatchShifter


class PreTraining(tf.keras.Model):
    '''
    Keras model for pre-training purpose.
    This model acts as a foundation model for downstream tasks.
    '''
    def __init__(
            self,
            nr_of_encoder_blocks,
            nr_of_heads,
            dropout_rate,
            encoder_ffn_units,
            embedding_dims,
            projection_head_units,
            reduced_dims,
            msk_rate,
            msk_scalar,
            nr_of_lookback_patches,
            nr_of_forecast_patches,
            mae_threshold,
            cl_threshold,
            pre_processor,
            **kwargs):
        '''
        args:
            nr_of_encoder_blocks (int):
                number of blocks of transformer encoders.
            nr_of_heads (int):
                number of attention heads of transformer encoders.
            dropout_rate (float):
                dropout rate.
            encoder_ffn_units (int):
                units of feed-forward networks of transformer encoders.
            embedding_dims (int): embedding dimension.
            projection_head_units (int):
                units of projection head of contrastive learning.
            reduced_dims (int):
                value of features dimension of a single patch.
            msk_rate (float): masking rate of the input patches.
            msk_scalar (float): values of the masked tokens.
            nr_of_lookback_patches (int):
                number of lookback patches.
            nr_of_forecast_patches (int):
                number of forecast patches.
            mae_threshold (float):
                stop criteria for masked autoencoder task.
            cl_threshold (float):
                stop criteria for contrastive learning task.
            pre_processor (tf.keras.Model):
                pre processor model from app_input_pipeline.
        '''
        super().__init__(**kwargs)

        self.margin = 0.10

        self.nr_of_lookback_patches = nr_of_lookback_patches
        self.nr_of_forecast_patches = nr_of_forecast_patches

        self.patch_masker = PatchMasker(
            masking_rate=msk_rate, msk_scalar=msk_scalar)

        self.patch_shifter = PatchShifter()

        self.encoder_representation = Representation(
            nr_of_encoder_blocks,
            nr_of_heads,
            dropout_rate,
            encoder_ffn_units,
            embedding_dims)

        self.lookback_forecast_concatter = tf.keras.layers.Concatenate(axis=1)

        self.decoder_tre = MppDecoder(
            reduced_dims,
            nr_of_lookback_patches + nr_of_forecast_patches,
            name='decoder_tre')
        self.decoder_sea = MppDecoder(
            reduced_dims,
            nr_of_lookback_patches + nr_of_forecast_patches,
            name='decoder_sea')
        self.decoder_res = MppDecoder(
            reduced_dims,
            nr_of_lookback_patches + nr_of_forecast_patches,
            name='decoder_res')

        self.projection_head = ProjectionHead(projection_head_units,
                                              name='projection_head')

        self.pre_processor = pre_processor

        self.mae_threshold = mae_threshold
        self.cl_threshold = cl_threshold

        # learning rate tracker
        self.lr_tracker = tf.keras.metrics.Mean(name='lr')
        # losses
        self.loss_tracker_mae = tf.keras.metrics.Mean(name='loss_mae')
        self.loss_tracker_cl = tf.keras.metrics.Mean(name='loss_cl')

        # metrics
        self.mae_tre = tf.keras.metrics.MeanAbsoluteError(name='mae_tre')
        self.mae_sea = tf.keras.metrics.MeanAbsoluteError(name='mae_sea')
        self.mae_res = tf.keras.metrics.MeanAbsoluteError(name='mae_res')
        self.cos_tre = tf.keras.metrics.CosineSimilarity(name='cos_tre')
        self.cos_sea = tf.keras.metrics.CosineSimilarity(name='cos_sea')
        self.cos_res = tf.keras.metrics.CosineSimilarity(name='cos_res')
        self.cos_true = tf.keras.metrics.CosineSimilarity(name='cos_true')
        self.cos_false = tf.keras.metrics.CosineSimilarity(name='cos_false')

        self.mae_composed = \
            tf.keras.metrics.MeanAbsoluteError(name='mae_composed')

        self.task_to_train = tf.Variable('mae')

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'pre_processor': self.pre_processor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config['pre_processor'] = tf.keras.layers.deserialize(
            config['pre_processor'])
        return cls(**config)

    def mask_patches(self, data):
        '''
            Masks both lookback and forecast patches.
            Patch steps to mask are determined randomly.
        '''
        x_tre, x_sea, x_res = data
        x_lb_tre = x_tre[:, :self.nr_of_lookback_patches]
        x_lb_sea = x_sea[:, :self.nr_of_lookback_patches]
        x_lb_res = x_res[:, :self.nr_of_lookback_patches]
        x_fc_tre = x_tre[:, self.nr_of_lookback_patches:]
        x_fc_sea = x_sea[:, self.nr_of_lookback_patches:]
        x_fc_res = x_res[:, self.nr_of_lookback_patches:]

        x_lb_sea_msk, x_lb_res_msk, x_lb_res_msk = self.patch_masker(
            (x_lb_tre, x_lb_sea, x_lb_res))
        x_fc_sea_msk, x_fc_res_msk, x_fc_res_msk = self.patch_masker(
            (x_fc_tre, x_fc_sea, x_fc_res))

        x_sea_msk = self.lookback_forecast_concatter(
            [x_lb_sea_msk, x_fc_sea_msk])
        x_res_msk = self.lookback_forecast_concatter(
            [x_lb_res_msk, x_fc_res_msk])
        x_res_msk = self.lookback_forecast_concatter(
            [x_lb_res_msk, x_fc_res_msk])

        return (x_sea_msk, x_res_msk, x_res_msk)

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
        x_tre, x_sea, x_res = data
        x_lb_tre = x_tre[:, :self.nr_of_lookback_patches]
        x_lb_sea = x_sea[:, :self.nr_of_lookback_patches]
        x_lb_res = x_res[:, :self.nr_of_lookback_patches]
        x_fc_tre = x_tre[:, self.nr_of_lookback_patches:]
        x_fc_sea = x_sea[:, self.nr_of_lookback_patches:]
        x_fc_res = x_res[:, self.nr_of_lookback_patches:]

        nr_of_forecast_patches = tf.shape(x_fc_tre)[1]

        # mask
        x_lb_sea_msk, x_lb_res_msk, x_lb_res_msk = self.patch_masker(
            (x_lb_tre, x_lb_sea, x_lb_res))
        x_sea_true = self.lookback_forecast_concatter(
            [x_lb_sea_msk, x_fc_tre])
        x_res_true = self.lookback_forecast_concatter(
            [x_lb_res_msk, x_fc_sea])
        x_res_true = self.lookback_forecast_concatter(
            [x_lb_res_msk, x_fc_res])

        # shift
        i = tf.random.uniform(
            shape=[],
            minval=1,
            maxval=nr_of_forecast_patches,
            dtype=tf.int32)
        x_fc_sea_sft, x_fc_res_sft, x_fc_res_sft = self.patch_shifter(
            (x_fc_tre, x_fc_sea, x_fc_res, i))
        x_sea_false = self.lookback_forecast_concatter(
            [x_lb_sea_msk, x_fc_sea_sft])
        x_res_false = self.lookback_forecast_concatter(
            [x_lb_res_msk, x_fc_res_sft])
        x_res_false = self.lookback_forecast_concatter(
            [x_lb_res_msk, x_fc_res_sft])

        return (x_sea_true,
                x_res_true,
                x_res_true,
                x_sea_false,
                x_res_false,
                x_res_false)

    @tf.function()
    def train_step(self, data):
        '''
        trains a step in two phases:
            1. masked patch prediction
            2. contrastive learning
        '''
        self.task_to_train.assign('mae')

        anchor_tre, anchor_sea, anchor_res, dates = data
        anchor_composed = \
            self.pre_processor.tre_denormalizer(anchor_tre) + \
            self.pre_processor.sea_denormalizer(anchor_sea) + \
            self.pre_processor.res_denormalizer(anchor_res)

        # masked auto-encoder (mae)
        msk_tre, msk_sea, msk_res = self.mask_patches(
            (anchor_tre, anchor_sea, anchor_res))

        with tf.GradientTape() as tape:
            y_pred_tre, y_pred_sea, y_pred_res = self(
                (msk_tre, msk_sea, msk_res, dates))

            pred_composed = \
                self.pre_processor.tre_denormalizer(y_pred_tre) + \
                self.pre_processor.sea_denormalizer(y_pred_sea) + \
                self.pre_processor.res_denormalizer(y_pred_res)

            # compute the loss value
            loss_mae = tf.keras.losses.mean_squared_error(
                y_pred=pred_composed, y_true=anchor_composed)

            trainable_vars = \
                self.encoder_representation.trainable_variables + \
                self.decoder_tre.trainable_variables + \
                self.decoder_sea.trainable_variables + \
                self.decoder_res.trainable_variables

        if tf.reduce_mean(loss_mae) > self.mae_threshold:

            gradients = tape.gradient(loss_mae, trainable_vars)

            # update weights
            self.mae_optimizer.apply_gradients(
                zip(gradients, trainable_vars))
        else:
            self.task_to_train.assign('cl')

        # compute own metrics
        self.loss_tracker_mae.update_state(loss_mae)
        self.mae_composed.update_state(
            y_pred=pred_composed,
            y_true=anchor_composed)
        self.mae_tre.update_state(y_pred=y_pred_tre, y_true=anchor_tre)
        self.mae_sea.update_state(y_pred=y_pred_sea, y_true=anchor_sea)
        self.mae_res.update_state(y_pred=y_pred_res, y_true=anchor_res)
        self.cos_tre.update_state(y_pred=y_pred_tre, y_true=anchor_tre)
        self.cos_sea.update_state(y_pred=y_pred_sea, y_true=anchor_sea)
        self.cos_res.update_state(y_pred=y_pred_res, y_true=anchor_res)

        # contrastive learning
        tre_true, sea_true, res_true, tre_false, sea_false, res_false = \
            self.augment_pairs((anchor_tre, anchor_sea, anchor_res))

        with tf.GradientTape() as tape:
            x_cont_temp_true = self.encoder_representation(
                (tre_true, sea_true, res_true, dates))
            x_cont_temp_false = self.encoder_representation(
                (tre_false, sea_false, res_false, dates))
            x_cont_temp_anchor = self.encoder_representation(
                (anchor_tre, anchor_sea, anchor_res, dates))

            y_logits_false = self.projection_head(x_cont_temp_false)
            y_logits_true = self.projection_head(x_cont_temp_true)
            y_logits_anchor = self.projection_head(x_cont_temp_anchor)

            # compute the loss value
            distance_true = tf.reduce_sum(
                tf.square(y_logits_anchor - y_logits_true), -1)
            distance_false = tf.reduce_sum(
                tf.square(y_logits_anchor - y_logits_false), -1)
            loss_cl = \
                tf.maximum(distance_true - distance_false + self.margin, 0.0)

        if self.task_to_train == 'cl':
            # compute gradients
            trainable_vars = \
                self.encoder_representation.trainable_variables + \
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
            'loss_mae': self.loss_tracker_mae.result(),
            'loss_cl': self.loss_tracker_cl.result(),
            'mae_tre': self.mae_tre.result(),
            'mae_sea': self.mae_sea.result(),
            'mae_res': self.mae_res.result(),
            'mae_composed': self.mae_composed.result(),
            'cos_tre': self.cos_tre.result(),
            'cos_sea': self.cos_sea.result(),
            'cos_res': self.cos_res.result(),
            'cos_true': self.cos_true.result(),
            'cos_false': self.cos_false.result(),
            'lr': self.lr_tracker.result()}

        return dic

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_tracker_mae,
            self.loss_tracker_cl,
            self.lr_tracker,
            self.mae_tre,
            self.mae_sea,
            self.mae_res,
            self.cos_tre,
            self.cos_sea,
            self.cos_res,
            self.cos_true,
            self.cos_false]

    def test_step(self, data):
        anchor_tre, anchor_sea, anchor_res, dates = data
        anchor_composed = \
            self.pre_processor.tre_denormalizer(anchor_tre) + \
            self.pre_processor.sea_denormalizer(anchor_sea) + \
            self.pre_processor.res_denormalizer(anchor_res)

        # mask the patches
        msk_tre, msk_sea, msk_res = self.mask_patches(
            (anchor_tre, anchor_sea, anchor_res))

        y_pred_tre, y_pred_sea, y_pred_res = \
            self((msk_tre, msk_sea, msk_res, dates))

        pred_composed = \
            self.pre_processor.tre_denormalizer(y_pred_tre) + \
            self.pre_processor.sea_denormalizer(y_pred_sea) + \
            self.pre_processor.res_denormalizer(y_pred_res)

        # compute the loss value
        loss_mae = tf.keras.losses.mean_squared_error(
            y_pred=pred_composed, y_true=anchor_composed)

        # compute own metrics
        self.loss_tracker_mae.update_state(loss_mae)
        self.mae_composed.update_state(
            y_pred=pred_composed,
            y_true=anchor_composed)
        self.mae_tre.update_state(y_pred=y_pred_tre, y_true=anchor_tre)
        self.mae_sea.update_state(y_pred=y_pred_sea, y_true=anchor_sea)
        self.mae_res.update_state(y_pred=y_pred_res, y_true=anchor_res)
        self.cos_tre.update_state(y_pred=y_pred_tre, y_true=anchor_tre)
        self.cos_sea.update_state(y_pred=y_pred_sea, y_true=anchor_sea)
        self.cos_res.update_state(y_pred=y_pred_res, y_true=anchor_res)

        # augment pairs
        tre_true, sea_true, res_true, tre_false, sea_false, res_false = \
            self.augment_pairs((anchor_tre, anchor_sea, anchor_res))

        x_cont_temp_true = \
            self.encoder_representation(
                (tre_true, sea_true, res_true, dates))
        x_cont_temp_false = \
            self.encoder_representation(
                (tre_false, sea_false, res_false, dates))
        x_cont_temp_anchor = self.encoder_representation(
            (anchor_tre, anchor_sea, anchor_res, dates))

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

        self.loss_tracker_cl.update_state(loss_cl)
        self.cos_true.update_state(
            y_true=y_logits_anchor, y_pred=y_logits_true)
        self.cos_false.update_state(
            y_true=y_logits_anchor, y_pred=y_logits_false)

        dic = {
            'loss_mae': self.loss_tracker_mae.result(),
            'loss_cl': self.loss_tracker_cl.result(),
            'mae_tre': self.mae_tre.result(),
            'mae_sea': self.mae_sea.result(),
            'mae_res': self.mae_res.result(),
            'mae_composed': self.mae_composed.result(),
            'cos_tre': self.cos_tre.result(),
            'cos_sea': self.cos_sea.result(),
            'cos_res': self.cos_res.result(),
            'cos_true': self.cos_true.result(),
            'cos_false': self.cos_false.result()}

        return dic

    def call(self, inputs):
        '''
        input: tuple of 4 arrays.
            1. dist: (none, timesteps, features)
            2. tre: (none, timesteps, features)
            3. sea: (none, timesteps, features)
            4. date: (none, features)
        '''
        y_cont_temp = self.encoder_representation(inputs)

        y_pred_tre = self.decoder_tre(y_cont_temp)
        y_pred_sea = self.decoder_sea(y_cont_temp)
        y_pred_res = self.decoder_res(y_cont_temp)

        return (y_pred_tre, y_pred_sea, y_pred_res)
