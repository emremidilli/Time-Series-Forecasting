import tensorflow as tf

from tsf_model.layers import PositionEmbedding, SingleStepDecoder


@tf.keras.saving.register_keras_serializable()
class FineTuning(tf.keras.Model):
    '''Keras model for fine-tuning purpose.'''
    def __init__(
            self,
            num_layers,
            hidden_dims,
            nr_of_heads,
            dff,
            dropout_rate,
            pre_trained_lookback_coefficient,
            msk_scalar,
            revIn_tre,
            revIn_sea,
            revIn_res,
            patch_tokenizer,
            encoder_representation,
            decoder_tre,
            decoder_sea,
            decoder_res,
            **kwargs):
        '''
        args:
            num_layers,
            hidden_dims,
            nr_of_heads,
            dff,
            dropout_rate,
            pre_trained_lookback_coefficient,
            msk_scalar,
            revIn_tre,
            revIn_sea,
            revIn_res,
            patch_tokenizer,
            encoder_representation,
            decoder_tre,
            decoder_sea,
            decoder_res,
        '''
        super().__init__(**kwargs)

        self.msk_scalar = msk_scalar
        self.pre_trained_lookback_coefficient = \
            pre_trained_lookback_coefficient

        self.pe = PositionEmbedding(embedding_dims=hidden_dims)

        self.revIn_tre = revIn_tre
        self.revIn_sea = revIn_sea
        self.revIn_res = revIn_res

        self.patch_tokenizer = patch_tokenizer
        self.encoder_representation = encoder_representation
        self.encoder_representation.trainable = False

        self.decoder_tre = decoder_tre
        self.decoder_sea = decoder_sea
        self.decoder_res = decoder_res

        self.decoder = SingleStepDecoder(
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            nr_of_heads=nr_of_heads,
            dff=dff,
            dropout_rate=dropout_rate)

        self.dense = tf.keras.layers.Dense(1)

        self.lb_fc_concatter = tf.keras.layers.Concatenate(axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'revIn_tre': self.revIn_tre,
                'revIn_sea': self.revIn_sea,
                'revIn_res': self.revIn_res,
                'patch_tokenizer': self.patch_tokenizer,
                'encoder_representation': self.encoder_representation,
                'decoder_tre': self.decoder_tre,
                'decoder_sea': self.decoder_sea,
                'decoder_res': self.decoder_res,
            }
        )
        return config

    def from_config(cls, config):
        config['revIn_tre'] = tf.keras.layers.deserialize(
            config['revIn_tre'])
        config['revIn_sea'] = tf.keras.layers.deserialize(
            config['revIn_sea'])
        config['revIn_res'] = tf.keras.layers.deserialize(
            config['revIn_res'])
        config['patch_tokenizer'] = tf.keras.layers.deserialize(
            config['patch_tokenizer'])
        config['encoder_representation'] = tf.keras.layers.deserialize(
            config['encoder_representation'])
        config['decoder_tre'] = tf.keras.layers.deserialize(
            config['decoder_tre'])
        config['decoder_sea'] = tf.keras.layers.deserialize(
            config['decoder_sea'])
        config['decoder_res'] = tf.keras.layers.deserialize(
            config['decoder_res'])

        return cls(**config)

    def call(self, inputs):
        '''
        Timesteps of forecast horizon are masked.
        args:
            tre: (None, timesteps, covariates)
            sea: (None, timesteps, covariates)
            res: (None, timesteps, covariates)
            dates: (None, features)
            shifted: (none, timesteps, covariates)
        returns:
            pred: (None, timesteps, covariates)
        '''
        lb_tre, lb_sea, lb_res, dates, shifted = inputs

        dimensions = tf.TensorShape(
            (
                lb_tre.shape[0],
                int(lb_tre.shape[1] / self.pre_trained_lookback_coefficient),
                lb_tre.shape[-1]
            ))
        fc_tre = tf.zeros(dimensions) + self.msk_scalar
        fc_sea = tf.zeros(dimensions) + self.msk_scalar
        fc_res = tf.zeros(dimensions) + self.msk_scalar

        tre = self.lb_fc_concatter([lb_tre, fc_tre])
        sea = self.lb_fc_concatter([lb_sea, fc_sea])
        res = self.lb_fc_concatter([lb_res, fc_res])

        # instance normalize
        tre_norm = self.revIn_tre(tre)
        sea_norm = self.revIn_sea(sea)
        res_norm = self.revIn_res(res)

        # tokenize timesteps into patches
        tre_patch = self.patch_tokenizer(tre_norm)
        sea_patch = self.patch_tokenizer(sea_norm)
        res_patch = self.patch_tokenizer(res_norm)

        y_cont_temp = self.encoder_representation(
            (tre_patch, sea_patch, res_patch, dates))

        y_pred_tre = self.decoder_tre(y_cont_temp)
        y_pred_sea = self.decoder_sea(y_cont_temp)
        y_pred_res = self.decoder_res(y_cont_temp)

        # instance denormalize
        y_pred_tre = self.revIn_tre.denormalize((tre, y_pred_tre))
        y_pred_sea = self.revIn_sea.denormalize((sea, y_pred_sea))
        y_pred_res = self.revIn_res.denormalize((res, y_pred_res))

        # compose
        t = y_pred_tre + y_pred_sea + y_pred_res

        z = self.pe(shifted)
        y = self.decoder((z, t))
        pred = self.dense(y)
        return pred
