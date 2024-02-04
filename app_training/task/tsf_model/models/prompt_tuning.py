import tensorflow as tf

from tsf_model.layers import PositionEmbedding


@tf.keras.saving.register_keras_serializable()
class PromptTuning(tf.keras.Model):
    '''Keras model for prompt-tuning purpose.'''
    def __init__(
            self,
            num_layers,
            hidden_dims,
            nr_of_heads,
            dff,
            dropout_rate,
            revIn_tre,
            revIn_sea,
            revIn_res,
            patch_tokenizer,
            **kwargs):
        '''
        args:

        '''
        super().__init__(**kwargs)

        self.pe = PositionEmbedding(embedding_dims=hidden_dims)

        self.revIn_tre = revIn_tre
        self.revIn_sea = revIn_sea
        self.revIn_res = revIn_res

        self.patch_tokenizer = patch_tokenizer

    def call(self, inputs):
        '''
        Only lookback window is taken as input.
        args:
            tre: (None, timesteps, covariates)
            sea: (None, timesteps, covariates)
            res: (None, timesteps, covariates)
            dates: (None, features)
        returns:
            pred: (None, timesteps, covariates)
        '''
        lb_tre, lb_sea, lb_res, dates = inputs

        # instance normalize
        tre_norm = self.revIn_tre(lb_tre)
        sea_norm = self.revIn_sea(lb_sea)
        res_norm = self.revIn_res(lb_res)

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
