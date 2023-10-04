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
            con_temp_pret,
            **kwargs):
        super().__init__(**kwargs)

        self.pe = PositionEmbedding(embedding_dims=hidden_dims)

        self.con_temp_pret = con_temp_pret

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
                'con_temp_pret': self.con_temp_pret,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config['con_temp_pret'] = tf.keras.layers.deserialize(
            config['con_temp_pret'])
        return cls(**config)

    def call(self, inputs):
        '''
        input: tuple of 4 arrays.
            1. dist: (none, timesteps, features)
            2. tre: (none, timesteps, features)
            3. sea: (none, timesteps, features)
            4. date: (none, features)
            5. shifted: (none, timesteps, 1)
            Timesteps of forecast horizon are masked.
        y: (none, timesteps, 1)
        '''
        dist, tre, sea, date, shifted = inputs
        t = self.con_temp_pret((dist, tre, sea, date))
        z = self.pe(shifted)
        y = self.decoder((z, t))
        y = self.dense(y)
        return y
