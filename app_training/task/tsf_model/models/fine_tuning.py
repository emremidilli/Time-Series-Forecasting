import tensorflow as tf

from tsf_model.layers import LinearHead


@tf.keras.saving.register_keras_serializable()
class FineTuning(tf.keras.Model):
    '''Keras model for fine-tuning multivariate time series.'''
    def __init__(
            self,
            revIn_tre,
            revIn_sea,
            revIn_res,
            patch_tokenizer,
            tre_embedding,
            sea_embedding,
            res_embedding,
            encoder_representation,
            nr_of_timesteps,
            nr_of_covariates,
            fine_tune_backbone,
            shared_prompt,
            decoder_tre,
            decoder_sea,
            decoder_res,
            **kwargs):
        '''
        args:

        '''
        super().__init__(**kwargs)

        self.nr_of_covariates = nr_of_covariates

        self.univariates = []
        for i in range(nr_of_covariates):
            self.univariates.append(
                Univariate(
                    revIn_tre=revIn_tre,
                    revIn_sea=revIn_sea,
                    revIn_res=revIn_res,
                    patch_tokenizer=patch_tokenizer,
                    tre_embedding=tre_embedding,
                    sea_embedding=sea_embedding,
                    res_embedding=res_embedding,
                    encoder_representation=encoder_representation,
                    nr_of_timesteps=nr_of_timesteps,
                    fine_tune_backbone=fine_tune_backbone,
                    shared_prompt=shared_prompt,
                    decoder_tre=decoder_tre,
                    decoder_sea=decoder_sea,
                    decoder_res=decoder_res))

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

        preds = []
        for i in range(self.nr_of_covariates):
            tre_i = tre[:, :, i]
            sea_i = sea[:, :, i]
            res_i = res[:, :, i]

            tre_i = tf.expand_dims(tre_i, axis=-1)
            sea_i = tf.expand_dims(sea_i, axis=-1)
            res_i = tf.expand_dims(res_i, axis=-1)

            univariate_model = self.univariates[i]

            pred_i = univariate_model((tre_i, sea_i, res_i, dates))

            preds.append(pred_i)

        pred = tf.stack(preds, axis=2)

        return pred

    def get_config(self):
        config = super().get_config()
        univariates_config = []
        for univariate_model in self.univariates:
            univariates_config.append(univariate_model.get_config())
        config.update({
            'nr_of_covariates': self.nr_of_covariates,
            'univariates': univariates_config
        })
        return config

    @classmethod
    def from_config(cls, config):
        univariates_config = config['univariates']
        config['univariates'] = \
            [Univariate.from_config(c) for c in univariates_config]
        return cls(**config)


@tf.keras.saving.register_keras_serializable()
class Univariate(tf.keras.Model):
    '''Keras model for fine-tuning univariate time series.'''
    def __init__(
            self,
            revIn_tre,
            revIn_sea,
            revIn_res,
            patch_tokenizer,
            tre_embedding,
            sea_embedding,
            res_embedding,
            encoder_representation,
            nr_of_timesteps,
            fine_tune_backbone,
            shared_prompt,
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

        self.tre_embedding = tre_embedding
        self.sea_embedding = sea_embedding
        self.res_embedding = res_embedding

        self.patch_tokenizer = patch_tokenizer
        self.encoder_representation = encoder_representation
        self.shared_prompt = shared_prompt

        self.nr_of_timesteps = nr_of_timesteps

        self.shared_prompt.trainable = False
        self.revIn_tre.trainable = False
        self.revIn_sea.trainable = False
        self.revIn_res.trainable = False
        self.decoder_tre.trainable = fine_tune_backbone
        self.decoder_sea.trainable = fine_tune_backbone
        self.decoder_res.trainable = fine_tune_backbone
        self.encoder_representation.trainable = fine_tune_backbone

        self.timesteps_concatter = tf.keras.layers.Concatenate(axis=1)

        self.lienar_head = LinearHead(
            nr_of_timesteps=nr_of_timesteps,
            nr_of_covariates=1,
            name='lienar_head')

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'revIn_tre': tf.keras.layers.serialize(self.revIn_tre),
                'revIn_sea': tf.keras.layers.serialize(self.revIn_sea),
                'revIn_res': tf.keras.layers.serialize(self.revIn_res),
                'tre_embedding': tf.keras.layers.serialize(self.tre_embedding),
                'sea_embedding': tf.keras.layers.serialize(self.sea_embedding),
                'res_embedding': tf.keras.layers.serialize(self.res_embedding),
                'patch_tokenizer': tf.keras.layers.serialize(
                    self.patch_tokenizer),
                'encoder_representation': tf.keras.layers.serialize(
                    self.encoder_representation),
                'shared_prompt': tf.keras.layers.serialize(self.shared_prompt),
                'decoder_tre': tf.keras.layers.serialize(self.decoder_tre),
                'decoder_sea': tf.keras.layers.serialize(self.decoder_sea),
                'decoder_res': tf.keras.layers.serialize(self.decoder_res)
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
        config['tre_embedding'] = tf.keras.layers.deserialize(
            config['tre_embedding'])
        config['sea_embedding'] = tf.keras.layers.deserialize(
            config['sea_embedding'])
        config['res_embedding'] = tf.keras.layers.deserialize(
            config['res_embedding'])
        config['patch_tokenizer'] = tf.keras.layers.deserialize(
            config['patch_tokenizer'])
        config['encoder_representation'] = tf.keras.layers.deserialize(
            config['encoder_representation'])
        config['shared_prompt'] = tf.keras.layers.deserialize(
            config['shared_prompt'])
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
            tre: (None, timesteps, 1)
            sea: (None, timesteps, 1)
            res: (None, timesteps, 1)
            dates: (None, features)
        returns:
            pred: (None, timesteps, 1)
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
        z = y_pred_tre + y_pred_sea + y_pred_res

        pred = self.lienar_head(z)

        return pred
