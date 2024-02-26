import tensorflow as tf

from tsf_model.layers import Representation, \
    LinearHead, ProjectionHead, PatchMasker, TimeStepShifter, \
    ReversibleInstanceNormalization, PatchTokenizer, \
    SoftPrompts


@tf.keras.saving.register_keras_serializable()
class PreTraining(tf.keras.Model):
    '''
    Keras model for pre-training purpose.
    This model acts as a foundation model for downstream tasks.
    '''
    def __init__(
            self,
            nr_of_covariates: int,
            patch_size: int,
            nr_of_encoder_blocks: int,
            nr_of_heads: int,
            dropout_rate: float,
            encoder_ffn_units: int,
            embedding_dims: int,
            projection_head_units: int,
            msk_scalar: float,
            nr_of_timesteps: int,
            contrastive_learning_patches: int,
            mae_threshold_comp: float,
            mae_threshold_tre: float,
            mae_threshold_sea: float,
            cl_threshold: float,
            cl_margin: float,
            pre_processor: tf.keras.Model,
            prompt_pool_size: int,
            nr_of_most_similar_prompts: int,
            **kwargs):
        '''
        args:
            nr_of_covariates: number of covariates.
            patch_size: number of timesteps in a patch.
            nr_of_encoder_blocks: number of blocks of transformer encoders.
            nr_of_heads: number of attention heads of transformer encoders.
            dropout_rate: dropout rate.
            encoder_ffn_units: units of feed-forward networks of
                transformer encoders.
            embedding_dims: embedding dimension.
            projection_head_units: units of projection head of
                contrastive learning.
            msk_scalar: values of the masked tokens.
            nr_of_timesteps: number of output timesteps.
            contrastive_learning_patches: number of patches for
                contrastive learning.
            mae_threshold_comp: stop criteria of composed value
                for masked autoencoder task.
            mae_threshold_tre: stop criteria of trend component for
                masked autoencoder task.
            mae_threshold_sea: stop criteria of seasonality component for
                masked autoencoder task.
            cl_threshold: stop criteria for contrastive learning task.
            cl_margin: margin for triple contrastive learning loss
            pre_processor: pre processor model from app_input_pipeline.
            prompt_pool_size: number of the prompts in prompt pool
        '''
        super(PreTraining, self).__init__(**kwargs)

        self.nr_of_covariates = nr_of_covariates
        self.patch_size = patch_size
        self.nr_of_encoder_blocks = nr_of_encoder_blocks
        self.nr_of_heads = nr_of_heads
        self.dropout_rate = dropout_rate
        self.encoder_ffn_units = encoder_ffn_units
        self.embedding_dims = embedding_dims
        self.projection_head_units = projection_head_units
        self.msk_scalar = msk_scalar
        self.nr_of_timesteps = nr_of_timesteps
        self.contrastive_learning_patches = contrastive_learning_patches
        self.mae_threshold_comp = mae_threshold_comp
        self.mae_threshold_tre = mae_threshold_tre
        self.mae_threshold_sea = mae_threshold_sea
        self.cl_threshold = cl_threshold
        self.cl_margin = cl_margin
        self.pre_processor = pre_processor
        self.prompt_pool_size = prompt_pool_size
        self.nr_of_most_similar_prompts = nr_of_most_similar_prompts

        self.revIn_tre = ReversibleInstanceNormalization(
            nr_of_covariates=nr_of_covariates,
            epsilon=1e-6)

        self.revIn_sea = ReversibleInstanceNormalization(
            nr_of_covariates=nr_of_covariates,
            epsilon=1e-6)

        self.revIn_res = ReversibleInstanceNormalization(
            nr_of_covariates=nr_of_covariates,
            epsilon=1e-6)

        self.patch_tokenizer = PatchTokenizer(
            patch_size=patch_size,
            nr_of_covariates=nr_of_covariates)

        self.shared_prompt = None

        self.patch_masker = PatchMasker(msk_scalar=msk_scalar)

        self.timestep_shifter = TimeStepShifter()

        self.tre_embedding = tf.keras.layers.Dense(units=self.embedding_dims)
        self.sea_embedding = tf.keras.layers.Dense(units=self.embedding_dims)
        self.res_embedding = tf.keras.layers.Dense(units=self.embedding_dims)

        self.encoder_representation = Representation(
            nr_of_encoder_blocks,
            nr_of_heads,
            dropout_rate,
            encoder_ffn_units,
            embedding_dims)

        self.timesteps_concatter = tf.keras.layers.Concatenate(axis=1)

        self.decoder_tre = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=nr_of_covariates,
            name='decoder_tre')
        self.decoder_sea = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=nr_of_covariates,
            name='decoder_sea')
        self.decoder_res = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=nr_of_covariates,
            name='decoder_res')

        self.projection_head = ProjectionHead(
            projection_head_units,
            name='projection_head')

        # learning rate tracker
        self.lr_tracker = tf.keras.metrics.Mean(name='lr')
        # losses
        self.loss_tracker_mae_comp = \
            tf.keras.metrics.Mean(name='loss_mae_comp')
        self.loss_tracker_mae_tre = \
            tf.keras.metrics.Mean(name='loss_mae_tre')
        self.loss_tracker_mae_sea = \
            tf.keras.metrics.Mean(name='loss_mae_sea')
        self.loss_tracker_cl = tf.keras.metrics.Mean(name='loss_cl')

        # metrics
        self.mae_tre = tf.keras.metrics.Mean(name='mae_tre')
        self.mae_sea = tf.keras.metrics.Mean(name='mae_sea')
        self.mae_res = tf.keras.metrics.Mean(name='mae_res')
        self.cos_tre = tf.keras.metrics.Mean(name='cos_tre')
        self.cos_sea = tf.keras.metrics.Mean(name='cos_sea')
        self.cos_res = tf.keras.metrics.Mean(name='cos_res')
        self.cos_true = tf.keras.metrics.CosineSimilarity(name='cos_true')
        self.cos_false = tf.keras.metrics.CosineSimilarity(name='cos_false')

        self.mae_composed = \
            tf.keras.metrics.Mean(name='mae_composed')

        self.mae_original = \
            tf.keras.metrics.Mean(name='mae_original')

        self.task_to_train = tf.Variable('mae')

        self.masks = tf.Variable(
            initial_value=False,
            trainable=False,
            dtype=tf.bool,
            name='masks')
        self.cl_masks = tf.Variable(
            initial_value=False,
            trainable=False,
            dtype=tf.bool,
            name='cl_masks')

    def compile(
            self,
            mae_comp_optimizer,
            mae_tre_optimizer,
            mae_sea_optimizer,
            cl_optimizer,
            **kwargs):
        super().compile(**kwargs)

        self.mae_comp_optimizer = mae_comp_optimizer
        self.mae_tre_optimizer = mae_tre_optimizer
        self.mae_sea_optimizer = mae_sea_optimizer
        self.cl_optimizer = cl_optimizer

    def get_compile_config(self):
        cfg = super().get_compile_config()
        cfg.update({
            'mae_comp_optimizer': self.mae_comp_optimizer,
            'mae_tre_optimizer': self.mae_tre_optimizer,
            'mae_sea_optimizer': self.mae_sea_optimizer,
            'cl_optimizer': self.cl_optimizer
        })

        return cfg

    def compile_from_config(self, config):
        mae_comp_optimizer = config['mae_comp_optimizer']
        mae_tre_optimizer = config['mae_tre_optimizer']
        mae_sea_optimizer = config['mae_sea_optimizer']
        cl_optimizer = config['cl_optimizer']

        self.compile(
            mae_comp_optimizer=mae_comp_optimizer,
            mae_tre_optimizer=mae_tre_optimizer,
            mae_sea_optimizer=mae_sea_optimizer,
            cl_optimizer=cl_optimizer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'pre_processor': tf.keras.saving.serialize_keras_object(
                    self.pre_processor),
            }
        )
        return config

    def from_config(cls, config):
        config['pre_processor'] = tf.keras.layers.deserialize(
            config['pre_processor'])
        return cls(**config)

    def calculate_masked_loss(
            self,
            y_pred,
            y_true,
            loss_fn):
        '''
        Calculates loss only for masked patches.
        y_true and y_pred are patched.
        based on masked patches, the loss is calculated.

        args:
            y_pred (None, timesteps, covariates) - predicted output
                which is unpactached.
            y_true (None, timesteps, covariates)- actual output
                which is unpatched.
            loss_fn (tf.keras.losses) - loss function

        returns
            loss (int) - calculated loss
        '''
        true_patched = self.patch_tokenizer(y_true)
        pred_patched = self.patch_tokenizer(y_pred)

        true_masked = tf.boolean_mask(
            tensor=true_patched,
            mask=self.masks,
            axis=1)
        pred_masked = tf.boolean_mask(
            tensor=pred_patched,
            mask=self.masks,
            axis=1)

        loss = loss_fn(y_pred=pred_masked, y_true=true_masked)

        return tf.reduce_mean(loss)

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
        x_lb_tre = x_tre[:, :self.contrastive_learning_patches]
        x_lb_sea = x_sea[:, :self.contrastive_learning_patches]
        x_lb_res = x_res[:, :self.contrastive_learning_patches]
        x_fc_tre = x_tre[:, self.contrastive_learning_patches:]
        x_fc_sea = x_sea[:, self.contrastive_learning_patches:]
        x_fc_res = x_res[:, self.contrastive_learning_patches:]

        nr_of_forecast_patches = tf.shape(x_fc_tre)[1]

        # mask
        x_fc_tre_msk, x_fc_sea_msk, x_fc_res_msk = self.patch_masker(
            (x_fc_tre, x_fc_sea, x_fc_res, self.cl_masks))

        x_tre_true = self.timesteps_concatter(
            [x_lb_tre, x_fc_tre_msk])
        x_sea_true = self.timesteps_concatter(
            [x_lb_sea, x_fc_sea_msk])
        x_res_true = self.timesteps_concatter(
            [x_lb_res, x_fc_res_msk])

        # shift
        i = tf.random.uniform(
            shape=[],
            minval=1,
            maxval=nr_of_forecast_patches,
            dtype=tf.int32)
        x_fc_tre_sft, x_fc_sea_sft, x_fc_res_sft = self.timestep_shifter(
            (x_fc_tre, x_fc_sea, x_fc_res, i))
        x_tre_false = self.timesteps_concatter(
            [x_lb_tre, x_fc_tre_sft])
        x_sea_false = self.timesteps_concatter(
            [x_lb_sea, x_fc_sea_sft])
        x_res_false = self.timesteps_concatter(
            [x_lb_res, x_fc_res_sft])

        return (x_tre_true,
                x_sea_true,
                x_res_true,
                x_tre_false,
                x_sea_false,
                x_res_false)

    @tf.function()
    def train_step(self, data):
        '''
        args:
            anchor_tre: (None, timesteps, covariates)
            anchor_sea: (None, timesteps, covariates)
            anchor_res: (None, timesteps, covariates)
            dates: (None, features)

        trains a step in two phases:
            1. masked patch prediction
            2. contrastive learning
        '''
        self.task_to_train.assign('mae')

        anchor_tre, anchor_sea, anchor_res, _ = data
        anchor_composed = anchor_tre + anchor_sea + anchor_res
        anchor_original = \
            self.pre_processor.data_denormalizer(anchor_composed)

        # masked auto-encoder (mae)
        y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed = \
            self(data)

        pred_original = \
            self.pre_processor.data_denormalizer(y_pred_composed)

        mae_tre = self.calculate_masked_loss(
            y_pred=y_pred_tre,
            y_true=anchor_tre,
            loss_fn=tf.keras.losses.mean_absolute_error)

        mae_sea = self.calculate_masked_loss(
            y_pred=y_pred_sea,
            y_true=anchor_sea,
            loss_fn=tf.keras.losses.mean_absolute_error)

        mae_comp = self.calculate_masked_loss(
            y_pred=y_pred_composed,
            y_true=anchor_composed,
            loss_fn=tf.keras.losses.mean_absolute_error)

        if mae_comp > self.mae_threshold_comp:
            with tf.GradientTape() as tape:
                y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed = \
                    self(data)

                pred_original = \
                    self.pre_processor.data_denormalizer(y_pred_composed)

                # compute the loss values
                loss_mae_comp = self.calculate_masked_loss(
                    y_pred=y_pred_composed,
                    y_true=anchor_composed,
                    loss_fn=tf.keras.losses.mean_squared_error)

                loss_mae_tre = self.calculate_masked_loss(
                    y_pred=y_pred_tre,
                    y_true=anchor_tre,
                    loss_fn=tf.keras.losses.mean_squared_error)

                loss_mae_sea = self.calculate_masked_loss(
                    y_pred=y_pred_sea,
                    y_true=anchor_sea,
                    loss_fn=tf.keras.losses.mean_squared_error)

                mae_trainable_vars = self.revIn_tre.trainable_variables + \
                    self.revIn_sea.trainable_variables + \
                    self.revIn_res.trainable_variables + \
                    self.encoder_representation.trainable_variables + \
                    self.decoder_tre.trainable_variables + \
                    self.decoder_sea.trainable_variables + \
                    self.decoder_res.trainable_variables

            # compute gradients
            mae_graidents = tape.gradient(
                loss_mae_comp,
                mae_trainable_vars)

            # update weights
            self.mae_comp_optimizer.apply_gradients(
                zip(mae_graidents, mae_trainable_vars))

            # log losses
            self.loss_tracker_mae_comp.update_state(loss_mae_comp)
            self.loss_tracker_mae_tre.update_state(loss_mae_tre)
            self.loss_tracker_mae_sea.update_state(loss_mae_sea)
        elif mae_tre > self.mae_threshold_tre:
            with tf.GradientTape() as tape:
                y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed = \
                    self(data)

                pred_original = \
                    self.pre_processor.data_denormalizer(y_pred_composed)

                # compute the loss values
                loss_mae_comp = self.calculate_masked_loss(
                    y_pred=y_pred_composed,
                    y_true=anchor_composed,
                    loss_fn=tf.keras.losses.mean_squared_error)

                loss_mae_tre = self.calculate_masked_loss(
                    y_pred=y_pred_tre,
                    y_true=anchor_tre,
                    loss_fn=tf.keras.losses.mean_squared_error)

                loss_mae_sea = self.calculate_masked_loss(
                    y_pred=y_pred_sea,
                    y_true=anchor_sea,
                    loss_fn=tf.keras.losses.mean_squared_error)

                mae_trainable_vars = self.revIn_tre.trainable_variables + \
                    self.encoder_representation.trainable_variables + \
                    self.decoder_tre.trainable_variables

            # compute gradients
            mae_graidents = tape.gradient(
                loss_mae_tre,
                mae_trainable_vars)

            # update weights
            self.mae_tre_optimizer.apply_gradients(
                zip(mae_graidents, mae_trainable_vars))

            # log losses
            self.loss_tracker_mae_comp.update_state(loss_mae_comp)
            self.loss_tracker_mae_tre.update_state(loss_mae_tre)
            self.loss_tracker_mae_sea.update_state(loss_mae_sea)

        elif mae_sea > self.mae_threshold_sea:
            with tf.GradientTape() as tape:
                y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed = \
                    self(data)

                pred_original = \
                    self.pre_processor.data_denormalizer(y_pred_composed)

                # compute the loss values
                # compute the loss values
                loss_mae_comp = self.calculate_masked_loss(
                    y_pred=y_pred_composed,
                    y_true=anchor_composed,
                    loss_fn=tf.keras.losses.mean_squared_error)

                loss_mae_tre = self.calculate_masked_loss(
                    y_pred=y_pred_tre,
                    y_true=anchor_tre,
                    loss_fn=tf.keras.losses.mean_squared_error)

                loss_mae_sea = self.calculate_masked_loss(
                    y_pred=y_pred_sea,
                    y_true=anchor_sea,
                    loss_fn=tf.keras.losses.mean_squared_error)

                mae_trainable_vars = self.revIn_sea.trainable_variables + \
                    self.encoder_representation.trainable_variables + \
                    self.decoder_sea.trainable_variables

            # compute gradients
            mae_graidents = tape.gradient(
                loss_mae_sea,
                mae_trainable_vars)

            # update weights
            self.mae_sea_optimizer.apply_gradients(
                zip(mae_graidents, mae_trainable_vars))

            # log losses
            self.loss_tracker_mae_comp.update_state(loss_mae_comp)
            self.loss_tracker_mae_tre.update_state(loss_mae_tre)
            self.loss_tracker_mae_sea.update_state(loss_mae_sea)

        else:
            self.task_to_train.assign('cl')

        mae_composed = self.calculate_masked_loss(
            y_pred=y_pred_composed,
            y_true=anchor_composed,
            loss_fn=tf.keras.losses.mean_absolute_error)

        mae_original = self.calculate_masked_loss(
            y_pred=pred_original,
            y_true=anchor_original,
            loss_fn=tf.keras.losses.mean_absolute_error)

        mae_tre = self.calculate_masked_loss(
            y_pred=y_pred_tre,
            y_true=anchor_tre,
            loss_fn=tf.keras.losses.mean_absolute_error)

        mae_sea = self.calculate_masked_loss(
            y_pred=y_pred_sea,
            y_true=anchor_sea,
            loss_fn=tf.keras.losses.mean_absolute_error)

        mae_res = self.calculate_masked_loss(
            y_pred=y_pred_res,
            y_true=anchor_res,
            loss_fn=tf.keras.losses.mean_absolute_error)

        cos_tre = self.calculate_masked_loss(
            y_pred=y_pred_tre,
            y_true=anchor_tre,
            loss_fn=tf.keras.losses.cosine_similarity)

        cos_sea = self.calculate_masked_loss(
            y_pred=y_pred_sea,
            y_true=anchor_sea,
            loss_fn=tf.keras.losses.cosine_similarity)

        cos_res = self.calculate_masked_loss(
            y_pred=y_pred_res,
            y_true=anchor_res,
            loss_fn=tf.keras.losses.cosine_similarity)

        # compute own metrics
        self.mae_composed.update_state(mae_composed)
        self.mae_original.update_state(mae_original)
        self.mae_tre.update_state(mae_tre)
        self.mae_sea.update_state(mae_sea)
        self.mae_res.update_state(mae_res)
        self.cos_tre.update_state(cos_tre)
        self.cos_sea.update_state(cos_sea)
        self.cos_res.update_state(cos_res)

        # contrastive learning
        with tf.GradientTape() as tape:
            y_logits_false, y_logits_true, y_logits_anchor = \
                self.call_contrastive_learning(data)

            # compute the loss value
            distance_true = tf.reduce_sum(
                tf.square(y_logits_anchor - y_logits_true), -1)
            distance_false = tf.reduce_sum(
                tf.square(y_logits_anchor - y_logits_false), -1)
            loss_cl = \
                tf\
                .maximum(distance_true - distance_false + self.cl_margin, 0.0)

        if self.task_to_train == 'cl':
            # compute gradients
            trainable_vars = \
                self.revIn_tre.trainable_variables + \
                self.revIn_sea.trainable_variables + \
                self.revIn_res.trainable_variables + \
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
            'loss_mae_comp': self.loss_tracker_mae_comp.result(),
            'loss_mae_tre': self.loss_tracker_mae_tre.result(),
            'loss_mae_sea': self.loss_tracker_mae_sea.result(),
            'loss_cl': self.loss_tracker_cl.result(),
            'mae_tre': self.mae_tre.result(),
            'mae_sea': self.mae_sea.result(),
            'mae_res': self.mae_res.result(),
            'mae_composed': self.mae_composed.result(),
            'mae_original': self.mae_original.result(),
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
            self.loss_tracker_mae_comp,
            self.loss_tracker_mae_tre,
            self.loss_tracker_mae_sea,
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
        anchor_composed = anchor_tre + anchor_sea + anchor_res

        anchor_original = \
            self.pre_processor.data_denormalizer(anchor_composed)

        y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed = \
            self(data)

        pred_original = \
            self.pre_processor.data_denormalizer(y_pred_composed)

        # compute the loss value
        loss_mae_comp = self.calculate_masked_loss(
            y_pred=y_pred_composed,
            y_true=anchor_composed,
            loss_fn=tf.keras.losses.mean_squared_error)

        loss_mae_tre = self.calculate_masked_loss(
            y_pred=y_pred_tre,
            y_true=anchor_tre,
            loss_fn=tf.keras.losses.mean_squared_error)

        loss_mae_sea = self.calculate_masked_loss(
            y_pred=y_pred_sea,
            y_true=anchor_sea,
            loss_fn=tf.keras.losses.mean_squared_error)

        # compute own metrics
        self.loss_tracker_mae_comp.update_state(loss_mae_comp)
        self.loss_tracker_mae_tre.update_state(loss_mae_tre)
        self.loss_tracker_mae_sea.update_state(loss_mae_sea)

        mae_composed = self.calculate_masked_loss(
            y_pred=y_pred_composed,
            y_true=anchor_composed,
            loss_fn=tf.keras.losses.mean_absolute_error)

        mae_original = self.calculate_masked_loss(
            y_pred=pred_original,
            y_true=anchor_original,
            loss_fn=tf.keras.losses.mean_absolute_error)

        mae_tre = self.calculate_masked_loss(
            y_pred=y_pred_tre,
            y_true=anchor_tre,
            loss_fn=tf.keras.losses.mean_absolute_error)

        mae_sea = self.calculate_masked_loss(
            y_pred=y_pred_sea,
            y_true=anchor_sea,
            loss_fn=tf.keras.losses.mean_absolute_error)

        mae_res = self.calculate_masked_loss(
            y_pred=y_pred_res,
            y_true=anchor_res,
            loss_fn=tf.keras.losses.mean_absolute_error)

        cos_tre = self.calculate_masked_loss(
            y_pred=y_pred_tre,
            y_true=anchor_tre,
            loss_fn=tf.keras.losses.cosine_similarity)

        cos_sea = self.calculate_masked_loss(
            y_pred=y_pred_sea,
            y_true=anchor_sea,
            loss_fn=tf.keras.losses.cosine_similarity)

        cos_res = self.calculate_masked_loss(
            y_pred=y_pred_res,
            y_true=anchor_res,
            loss_fn=tf.keras.losses.cosine_similarity)

        self.mae_composed.update_state(mae_composed)
        self.mae_original.update_state(mae_original)
        self.mae_tre.update_state(mae_tre)
        self.mae_sea.update_state(mae_sea)
        self.mae_res.update_state(mae_res)
        self.cos_tre.update_state(cos_tre)
        self.cos_sea.update_state(cos_sea)
        self.cos_res.update_state(cos_res)

        # augment pairs
        y_logits_false, y_logits_true, y_logits_anchor = \
            self.call_contrastive_learning(data)

        # compute the loss value
        distance_true = tf.reduce_sum(
            tf.square(y_logits_anchor - y_logits_true), -1)
        distance_false = tf.reduce_sum(
            tf.square(y_logits_anchor - y_logits_false), -1)
        loss_cl = tf.maximum(
            distance_true - distance_false + self.cl_margin, 0.0)

        self.loss_tracker_cl.update_state(loss_cl)
        self.cos_true.update_state(
            y_true=y_logits_anchor, y_pred=y_logits_true)
        self.cos_false.update_state(
            y_true=y_logits_anchor, y_pred=y_logits_false)

        dic = {
            'loss_mae_tre': self.loss_tracker_mae_tre.result(),
            'loss_mae_sea': self.loss_tracker_mae_sea.result(),
            'loss_mae_comp': self.loss_tracker_mae_comp.result(),
            'loss_cl': self.loss_tracker_cl.result(),
            'mae_tre': self.mae_tre.result(),
            'mae_sea': self.mae_sea.result(),
            'mae_res': self.mae_res.result(),
            'mae_composed': self.mae_composed.result(),
            'mae_original': self.mae_original.result(),
            'cos_tre': self.cos_tre.result(),
            'cos_sea': self.cos_sea.result(),
            'cos_res': self.cos_res.result(),
            'cos_true': self.cos_true.result(),
            'cos_false': self.cos_false.result()}

        return dic

    def call_contrastive_learning(self, inputs):
        '''
        args:
            inputs:
                tre: (none, timesteps, covariates)
                sea: (none, timesteps, covariates)
                res: (none, timesteps, covariates)
                dates: (none, features)
        '''

        tre, sea, res, dates = inputs

        # instance normalize
        tre_norm = self.revIn_tre(tre)
        sea_norm = self.revIn_sea(sea)
        res_norm = self.revIn_res(res)

        # tokenize timesteps into patches
        tre_patch = self.patch_tokenizer(tre_norm)
        sea_patch = self.patch_tokenizer(sea_norm)
        res_patch = self.patch_tokenizer(res_norm)

        if self.shared_prompt is None:
            self.shared_prompt = SoftPrompts(
                key_dims=self.nr_of_timesteps * self.nr_of_covariates,
                embedding_dims=self.embedding_dims,
                prompt_length=tre_patch.shape[-2],
                prompt_pool_size=self.prompt_pool_size,
                nr_of_most_similar_prompts=self.nr_of_most_similar_prompts)

        tre_prompts = self.shared_prompt(tre_norm)
        sea_prompts = self.shared_prompt(sea_norm)
        res_prompts = self.shared_prompt(res_norm)

        tre_true, sea_true, res_true, tre_false, sea_false, res_false = \
            self.augment_pairs((tre_patch, sea_patch, res_patch))\

        tre_anchor_embed = self.tre_embedding(tre_patch)
        tre_true_embed = self.tre_embedding(tre_true)
        tre_false_embed = self.tre_embedding(tre_false)

        sea_anchor_embed = self.sea_embedding(sea_patch)
        sea_true_embed = self.sea_embedding(sea_true)
        sea_false_embed = self.sea_embedding(sea_false)

        res_anchor_embed = self.res_embedding(res_patch)
        res_true_embed = self.res_embedding(res_true)
        res_false_embed = self.res_embedding(res_false)

        tre_anchor_embed = \
            self.timesteps_concatter([tre_prompts, tre_anchor_embed])
        sea_anchor_embed = \
            self.timesteps_concatter([sea_prompts, sea_anchor_embed])
        res_anchor_embed = \
            self.timesteps_concatter([res_prompts, res_anchor_embed])

        tre_false_embed = \
            self.timesteps_concatter([tre_prompts, tre_false_embed])
        sea_false_embed = \
            self.timesteps_concatter([sea_prompts, sea_false_embed])
        res_false_embed = \
            self.timesteps_concatter([res_prompts, res_false_embed])

        tre_true_embed = \
            self.timesteps_concatter([tre_prompts, tre_true_embed])
        sea_true_embed = \
            self.timesteps_concatter([sea_prompts, sea_true_embed])
        res_true_embed = \
            self.timesteps_concatter([res_prompts, res_true_embed])

        x_cont_temp_true = self.encoder_representation(
            (tre_true_embed, sea_true_embed, res_true_embed, dates))
        x_cont_temp_false = self.encoder_representation(
            (tre_false_embed, sea_false_embed, res_false_embed, dates))
        x_cont_temp_anchor = self.encoder_representation(
            (tre_anchor_embed, sea_anchor_embed, res_anchor_embed, dates))

        y_logits_false = self.projection_head(x_cont_temp_false)
        y_logits_true = self.projection_head(x_cont_temp_true)
        y_logits_anchor = self.projection_head(x_cont_temp_anchor)

        return y_logits_false, y_logits_true, y_logits_anchor

    def call(self, inputs):
        '''
        args:
            tre: (None, timesteps, covariates)
            sea: (None, timesteps, covariates)
            res: (None, timesteps, covariates)
            dates: (None, features)
        returns:
            y_pred_tre: (None, timesteps, covariates)
            y_pred_sea: (None, timesteps, covariates),
            y_pred_res: (None, timesteps, covariates)
            y_pred_composed: (None, timesteps, covariates)
        '''

        tre, sea, res, dates = inputs

        # instance normalize
        tre_norm = self.revIn_tre(tre)
        sea_norm = self.revIn_sea(sea)
        res_norm = self.revIn_res(res)

        # tokenize timesteps into patches
        tre_patch = self.patch_tokenizer(tre_norm)
        sea_patch = self.patch_tokenizer(sea_norm)
        res_patch = self.patch_tokenizer(res_norm)

        tre_patch, sea_patch, res_patch = self.patch_masker(
            (tre_patch, sea_patch, res_patch, self.masks))

        if self.shared_prompt is None:
            self.shared_prompt = SoftPrompts(
                key_dims=self.nr_of_timesteps * self.nr_of_covariates,
                embedding_dims=self.embedding_dims,
                prompt_length=tre_patch.shape[-2],
                prompt_pool_size=self.prompt_pool_size,
                nr_of_most_similar_prompts=self.nr_of_most_similar_prompts)

        tre_prompts = self.shared_prompt(tre_norm)
        sea_prompts = self.shared_prompt(sea_norm)
        res_prompts = self.shared_prompt(res_norm)

        tre_embed = self.tre_embedding(tre_patch)
        sea_embed = self.sea_embedding(sea_patch)
        res_embed = self.res_embedding(res_patch)

        tre_input = self.timesteps_concatter([tre_prompts, tre_embed])
        sea_input = self.timesteps_concatter([sea_prompts, sea_embed])
        res_input = self.timesteps_concatter([res_prompts, res_embed])

        y_cont_temp = self.encoder_representation(
            (tre_input, sea_input, res_input, dates))

        y_pred_tre = self.decoder_tre(y_cont_temp)
        y_pred_sea = self.decoder_sea(y_cont_temp)
        y_pred_res = self.decoder_res(y_cont_temp)

        # instance denormalize
        y_pred_tre = self.revIn_tre.denormalize((tre, y_pred_tre))
        y_pred_sea = self.revIn_sea.denormalize((sea, y_pred_sea))
        y_pred_res = self.revIn_res.denormalize((res, y_pred_res))

        # compose
        y_pred_composed = y_pred_tre + y_pred_sea + y_pred_res

        return (y_pred_tre, y_pred_sea, y_pred_res, y_pred_composed)
