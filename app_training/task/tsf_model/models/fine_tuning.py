import tensorflow as tf

from tsf_model.layers import PositionEmbedding, LinearHead


@tf.keras.saving.register_keras_serializable()
class FineTuning(tf.keras.Model):
    '''Keras model for fine-tuning purpose.'''
    def __init__(
            self,
            revIn_tre,
            revIn_sea,
            revIn_res,
            patch_tokenizer,
            encoder_representation,
            nr_of_timesteps,
            nr_of_covariates,
            fine_tune_backbone,
            decoder_tre,
            decoder_sea,
            decoder_res,
            **kwargs):
        '''
        args:

        '''
        super().__init__(**kwargs)

        self.revIn_tre = revIn_tre
        self.revIn_sea = revIn_sea
        self.revIn_res = revIn_res
        self.decoder_tre = decoder_tre
        self.decoder_sea = decoder_sea
        self.decoder_res = decoder_res
        self.fine_tune_backbone = fine_tune_backbone

        self.patch_tokenizer = patch_tokenizer
        self.encoder_representation = encoder_representation

        self.nr_of_timesteps = nr_of_timesteps

        self.revIn_tre.trainable = False
        self.revIn_sea.trainable = False
        self.revIn_res.trainable = False
        self.decoder_tre.trainable = fine_tune_backbone
        self.decoder_sea.trainable = fine_tune_backbone
        self.decoder_res.trainable = fine_tune_backbone
        self.encoder_representation.trainable = fine_tune_backbone

        self.lienar_head = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=nr_of_covariates,
            name='lienar_head')

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
                'decoder_res': self.decoder_res
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
        returns:
            pred: (None, timesteps, covariates)
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
        z = y_pred_tre + y_pred_sea + y_pred_res

        pred = self.lienar_head(z)

        return pred
