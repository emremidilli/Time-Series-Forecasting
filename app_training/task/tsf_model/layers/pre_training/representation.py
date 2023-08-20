from . import PositionEmbedding, TransformerEncoder, VariableSelection

import tensorflow as tf


class Representation(tf.keras.layers.Layer):
    def __init__(
            self,
            iNrOfEncoderBlocks,
            iNrOfHeads,
            fDropoutRate,
            iEncoderFfnUnits,
            iEmbeddingDims,
            **kwargs):

        super().__init__(**kwargs)

        self.pe_dist_temporal = PositionEmbedding(iUnits=iEmbeddingDims,
                                                  name='pe_dist_temporal')
        self.pe_tre_temporal = PositionEmbedding(iUnits=iEmbeddingDims,
                                                 name='pe_tre_temporal')
        self.pe_sea_temporal = PositionEmbedding(iUnits=iEmbeddingDims,
                                                 name='pe_sea_temporal')

        self.temporal_to_contextual = tf.keras.layers.Permute((2, 1))

        self.pe_dist_contextual = PositionEmbedding(iUnits=iEmbeddingDims,
                                                    name='pe_dist_contextual')
        self.pe_tre_contextual = PositionEmbedding(iUnits=iEmbeddingDims,
                                                   name='pe_tre_contextual')
        self.pe_sea_contextual = PositionEmbedding(iUnits=iEmbeddingDims,
                                                   name='pe_tre_contextual')

        self.concat_contextuals = tf.keras.layers.Concatenate(axis=1)

        self.variable_selection_single_step = VariableSelection(
            num_features=3,
            units=iEmbeddingDims,
            dropout_rate=fDropoutRate,
            name='variable_selection_single_step')
        self.variable_selection_temporals = tf.keras.layers.TimeDistributed(
            self.variable_selection_single_step)

        self.encoders_temporal = []
        for i in range(iNrOfEncoderBlocks):
            self.encoders_temporal.append(
                TransformerEncoder(
                    embed_dim=iEmbeddingDims,
                    num_heads=iNrOfHeads,
                    feedforward_dim=iEncoderFfnUnits,
                    dropout_rate=fDropoutRate,
                    name=f'encoders_temporal{i}'
                )
            )

        self.encoders_contextual = []
        for i in range(iNrOfEncoderBlocks):
            self.encoders_contextual.append(
                TransformerEncoder(
                    embed_dim=iEmbeddingDims,
                    num_heads=iNrOfHeads,
                    feedforward_dim=iEncoderFfnUnits,
                    dropout_rate=fDropoutRate,
                    name=f'encoders_contextual{i}'
                )
            )

        self.concat_temporal_contextual = tf.keras.layers.Concatenate(axis=1)

        self.encoders_cont_temp = []
        for i in range(iNrOfEncoderBlocks):
            self.encoders_cont_temp.append(
                TransformerEncoder(
                    embed_dim=iEmbeddingDims,
                    num_heads=iNrOfHeads,
                    feedforward_dim=iEncoderFfnUnits,
                    dropout_rate=fDropoutRate,
                    name=f'encoders_cont_temp{i}'
                )
            )

    def call(self, x):
        '''
        inputs: tuple of 3 elements
            distribution - tuple of 3 elements (None, timesteps, feature)
            trend - tuple of 3 elements (None, timesteps, feature)
            seasonality - tuple of 3 elements (None, timesteps, feature)
        '''

        x_dist_temp, x_tre_temp, x_sea_temp = x

        x_dist_cont = self.temporal_to_contextual(x_dist_temp)
        x_tre_cont = self.temporal_to_contextual(x_tre_temp)
        x_sea_cont = self.temporal_to_contextual(x_sea_temp)

        x_dist_temp = self.pe_dist_temporal(x_dist_temp)
        x_tre_temp = self.pe_tre_temporal(x_tre_temp)
        x_sea_temp = self.pe_sea_temporal(x_sea_temp)

        x_dist_cont = self.pe_dist_contextual(x_dist_cont)
        x_tre_cont = self.pe_tre_contextual(x_tre_cont)
        x_sea_cont = self.pe_sea_contextual(x_sea_cont)

        x_temp = self.variable_selection_temporals(
            [x_dist_temp, x_tre_temp, x_sea_temp])
        x_cont = self.concat_contextuals(
            [x_dist_cont, x_tre_cont, x_sea_cont])

        for encoder in self.encoders_temporal:
            x_temp = encoder(x_temp)

        for encoder in self.encoders_contextual:
            x_cont = encoder(x_cont)

        x_cont_temp = self.concat_temporal_contextual([x_temp, x_cont])

        for encoder in self.encoders_cont_temp:
            x_cont_temp = encoder(x_cont_temp)

        return x_cont_temp