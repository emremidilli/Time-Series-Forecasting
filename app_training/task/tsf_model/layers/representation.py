from . import PositionEmbedding, TransformerEncoder, Time2Vec

import tensorflow as tf


class Representation(tf.keras.layers.Layer):
    '''Encoder representation layer.'''
    def __init__(
            self,
            nr_of_encoder_blocks,
            nr_of_heads,
            dropout_rate,
            encoder_ffn_units,
            embedding_dims,
            **kwargs):

        super().__init__(**kwargs)

        self.pe_tre_temporal = PositionEmbedding(
            embedding_dims=embedding_dims,
            name='pe_tre_temporal')
        self.pe_sea_temporal = PositionEmbedding(
            embedding_dims=embedding_dims,
            name='pe_sea_temporal')
        self.pe_res_temporal = PositionEmbedding(
            embedding_dims=embedding_dims,
            name='pe_res_temporal')

        self.time2vec = Time2Vec(embedding_dims=embedding_dims,
                                 name='time2vec')

        self.concat_temporals = tf.keras.layers.Concatenate(axis=1)

        self.encoders_temporal = []
        for i in range(nr_of_encoder_blocks):
            self.encoders_temporal.append(
                TransformerEncoder(
                    embed_dim=embedding_dims,
                    num_heads=nr_of_heads,
                    feedforward_dim=encoder_ffn_units,
                    dropout_rate=dropout_rate,
                    name=f'encoders_temporal{i}'
                ))

    def call(self, x):
        '''
        inputs: tuple of 4 elements
            trend: (None, timesteps, features)
            seasonality: (None, timesteps, features)
            residual: (None, timesteps, features)
            dates: (None, features)
        '''

        x_tre_temp, x_sea_temp, x_res_temp, x_dates = x

        x_tre_temp = self.pe_tre_temporal(x_tre_temp)
        x_sea_temp = self.pe_sea_temporal(x_sea_temp)
        x_res_temp = self.pe_res_temporal(x_res_temp)

        x_dates = self.time2vec(x_dates)

        x_temp = self.concat_temporals(
            [x_tre_temp, x_sea_temp, x_res_temp, x_dates])

        for encoder in self.encoders_temporal:
            x_temp = encoder(x_temp)

        return x_temp
