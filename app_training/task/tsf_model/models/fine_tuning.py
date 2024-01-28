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
            pre_trained_model,
            **kwargs):
        super().__init__(**kwargs)

        self.pe = PositionEmbedding(embedding_dims=hidden_dims)

        self.pre_trained_model = pre_trained_model

        self.decoder = SingleStepDecoder(
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            nr_of_heads=nr_of_heads,
            dff=dff,
            dropout_rate=dropout_rate)

        self.dense = tf.keras.layers.Dense(1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'pre_trained_model': self.pre_trained_model,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config['pre_trained_model'] = tf.keras.layers.deserialize(
            config['pre_trained_model'])
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
        tre, sea, res, date, shifted = inputs
        t = self.pre_trained_model((tre, sea, res, date))
        z = self.pe(shifted)
        y = self.decoder((z, t))
        pred = self.dense(y)
        return pred
