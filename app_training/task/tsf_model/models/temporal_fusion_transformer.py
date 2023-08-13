import sys
sys.path.append( '../')
from layers.temporal_fusion_transformer.variable_selection_network import variable_selection_network
from layers.temporal_fusion_transformer.interpretable_multi_head_attention import interpretable_multi_head_attention, get_decoder_mask
from layers.temporal_fusion_transformer.gated_residual_network import gated_residual_network
from layers.temporal_fusion_transformer.gated_linear_unit import gated_linear_unit

import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

class temporal_fusion_transformer(tf.keras.Model):

    def __init__(self,
                 iNrOfLookbackPatches,
                 iNrOfForecastPatches,
                 aTargetQuantiles,
                 fDropout,
                 iModelDims,
                 iNrOfChannels,
                 **kwargs):
        super().__init__(**kwargs)


        self.iNrOfLookbackPatches = iNrOfLookbackPatches
        self.iNrOfForecastPatches = iNrOfForecastPatches
        self.aTargetQuantiles = aTargetQuantiles
        self.fDropout = fDropout
        self.iModelDims = iModelDims
        self.iNrOfChannels = iNrOfChannels

        self.oStaticEncoder = tf.keras.layers.Dense(units = iModelDims)

        self.oVsnStatic = variable_selection_network(
            iModelDims = iModelDims,
            iNrOfVariables = 2 * iNrOfChannels, # one for static digit, one for transition.`
            fDropout = fDropout,
            bIsWithExternal = False
        )

        self.oLookbackRepeat = tf.keras.layers.RepeatVector(n = iNrOfLookbackPatches)
        self.oForecastRepeat = tf.keras.layers.RepeatVector(n = iNrOfForecastPatches)
        self.oAllRepeat = tf.keras.layers.RepeatVector(n = iNrOfLookbackPatches+ iNrOfForecastPatches)

        # static covariate encoders to be used in different parts of the tft model
        self.oStaticContextTemporalVsn = gated_residual_network(
            iInputDims = iModelDims ,
            iOutputDims = iModelDims,
            fDropout = fDropout,
            bIsWithStaticCovariate=False
        )

        self.oStaticContextStateHidden = gated_residual_network(
            iInputDims = iModelDims ,
            iOutputDims = iModelDims,
            fDropout = fDropout,
            bIsWithStaticCovariate=False
        )

        self.oStaticContextStateCarry = gated_residual_network(
            iInputDims = iModelDims ,
            iOutputDims = iModelDims,
            fDropout = fDropout,
            bIsWithStaticCovariate=False
        )

        self.oStaticContextEnrichment = gated_residual_network(
            iInputDims = iModelDims ,
            iOutputDims = iModelDims,
            fDropout = 0,
            bIsWithStaticCovariate=False
        )

        # temporal variable selection networks.
        # lookback time patches share weights for variable_selection_network accross each other.
        # similarly, forecast time patches do so as well.
        self. oVsnLookback = variable_selection_network(
            iModelDims = iModelDims,
            iNrOfVariables = 6 * iNrOfChannels,
            fDropout = fDropout,
            bIsWithExternal = True
        )
        self.oTimeDistVsnLookback = tf.keras.layers.TimeDistributed(self.oVsnLookback)


        self.oVsnForecast = variable_selection_network(
            iModelDims = iModelDims,
            iNrOfVariables = 6 * iNrOfChannels,
            fDropout = fDropout,
            bIsWithExternal = True
        )
        self.oTimeDistVsnForecast = tf.keras.layers.TimeDistributed(self.oVsnForecast)


        # used for encoding and decoding with LSTMs.
        self.oLstmEncoder = tf.keras.layers.LSTM(
            units = iModelDims,
            return_sequences=True,
            return_state=True
        )

        self.oLstmDecoder = tf.keras.layers.LSTM(
            units = iModelDims,
            return_sequences=True,
            return_state=True
        )

        # used for concating lookback and forecast outputs
        self.oConcatter = tf.keras.layers.Concatenate(axis = 1)

        # gate, add & norm
        self.oGates_1 = tf.keras.layers.TimeDistributed(
            gated_linear_unit(iFfnUnits = iModelDims)
        )
        self.oLayerNorm_1 =  tf.keras.layers.TimeDistributed(
            tf.keras.layers.LayerNormalization()
        )


        # static enrichment
        self.oStaticEnrichment  = tf.keras.layers.TimeDistributed(
            gated_residual_network(
                iInputDims = iModelDims ,
                iOutputDims = iModelDims,
                fDropout = fDropout,
                bIsWithStaticCovariate=True
            )
        )


        # temporal self-attention -> masked interpretable multi-head attention
        self.oInterpretableMha = interpretable_multi_head_attention(
            iNrOfHeads = 2,
            iModelDims = iModelDims,
            fDropout = fDropout
        )


        # temporal self-attention -> gate, add & norm
        self.oGates_2 = tf.keras.layers.TimeDistributed(
            gated_linear_unit(iFfnUnits = iModelDims)
        )
        self.oLayerNorm_2 =  tf.keras.layers.TimeDistributed(
            tf.keras.layers.LayerNormalization()
        )


        # position-wise feed-forward
        self.oPositionWiseFeedForward  = tf.keras.layers.TimeDistributed(
            gated_residual_network(
                iInputDims = iModelDims ,
                iOutputDims = iModelDims,
                fDropout = fDropout,
                bIsWithStaticCovariate=False
            )
        )


        # output -> gate, add & norm
        self.oGates_3 = tf.keras.layers.TimeDistributed(
            gated_linear_unit(iFfnUnits = iModelDims)
        )
        self.oLayerNorm_3 =  tf.keras.layers.TimeDistributed(
            tf.keras.layers.LayerNormalization()
        )


        # output -> dense
        self.oDenseQuantile =  tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units = (len(aTargetQuantiles) * iNrOfChannels))
        )
        self.oReshape = tf.keras.layers.Reshape(target_shape = (-1, len(aTargetQuantiles), iNrOfChannels))



    def call(self, inputs):
        '''
        Inference for TFT model.
        Args:
            inputs - tuple of 3 elements:
                x_lookback - (None, iNrOfLookbackPatches, iModelDims ,6 x iNrOfChannels) are the representations of each channel within model_dims for each lookback patches.
                    6: {dist, tre, sea, tic, known, observed}
                x_forecast - (None, iNrOfForecastPatches, iModelDims ,6 x iNrOfChannels) are the representations of each channel within model_dims for each lookback patches.
                    6: {dist, tre, sea, tic, known, observed}
                x_static - (None, 1, 2 x iNrOfChannels) are the static representations of each channel.
                    2: {static digit, transition}
        '''

        x_lookback, x_forecast , x_static = inputs
        s = self.oStaticEncoder(x_static)
        s_c, s_v = self.oVsnStatic([s, None])

        # producing static vectors via static covariate encoders
        s_c_temporal_vsn = self.oStaticContextTemporalVsn([s_c, None])
        s_c_static_enrichment = self.oStaticContextEnrichment([s_c, None])
        s_c_static_state_hidden = self.oStaticContextStateHidden([s_c, None])
        s_c_static_state_carry = self.oStaticContextStateCarry([s_c, None])

        # variable selection for temporal steps
        s_c_lookback_vsn = self.oLookbackRepeat(s_c_temporal_vsn)
        s_c_forecast_vsn = self.oForecastRepeat(s_c_temporal_vsn)

        y_lookback, w_lookback = self.oTimeDistVsnLookback([x_lookback, s_c_lookback_vsn])
        y_forecast, w_forecast = self.oTimeDistVsnForecast([x_forecast, s_c_forecast_vsn])

        # encode & decode with LSTMs
        y_encoder, h_encoder, c_encoder = self.oLstmEncoder(
            y_lookback,
            initial_state=[s_c_static_state_hidden,s_c_static_state_carry]
        )
        y_decoder, h_decoder, c_decoder = self.oLstmDecoder(
            y_forecast,
            initial_state=[h_encoder,c_encoder]
        )

        # concat outputs
        y_lookback_forecast = self.oConcatter([y_lookback, y_forecast])
        y_encoder_decoder = self.oConcatter([y_encoder, y_decoder])

        # gate, add & norm
        y_tft_encoder = self.oGates_1(y_encoder_decoder)
        y_tft_encoder = y_tft_encoder + y_lookback_forecast
        y_tft_encoder = self.oLayerNorm_1(y_tft_encoder)


        # static enrichment
        s_c_static_enrichment = self.oAllRepeat(s_c_static_enrichment)
        y_static_enrichment = self.oStaticEnrichment(
            [y_tft_encoder, s_c_static_enrichment]
        )

        # temporal self-attention
        aDecoderMask = get_decoder_mask(y_static_enrichment)
        y_attention, w_attention = self.oInterpretableMha(
            q = y_static_enrichment,
            k = y_static_enrichment,
            v = y_static_enrichment,
            mask = aDecoderMask
        )

        # gating, add & norm
        y_tft_self_attn = self.oGates_2(y_attention)
        y_tft_self_attn = y_tft_self_attn + y_static_enrichment
        y_tft_self_attn = self.oLayerNorm_2(y_tft_self_attn)


        # position-wise feed-forward

        y_ffn = self.oPositionWiseFeedForward(
            [y_tft_self_attn]
        )


        # Output -> gating, add & norm
        y = self.oGates_3(y_ffn)
        y = y + y_tft_encoder
        y = self.oLayerNorm_3(y)

        # Output only forecasting patches
        y = y[:, -self.iNrOfForecastPatches:, Ellipsis]

        # Output - dense
        y = self.oDenseQuantile(y)
        y = self.oReshape(y)

        return y



    def quantile_loss(self, true, pred):
        '''
        Returns quantile loss for specified quantiles.
        Args:
            true: Targets (None, nr_of_forecast_patches, nr_of_quantiles, nr_of_channels)
            pred: Predictions (None, nr_of_forecast_patches, nr_of_quantiles, nr_of_channels)
        '''
        loss = 0.
        for i, quantile in enumerate(self.aTargetQuantiles):
            for j in range(self.iNrOfChannels):
                prediction_underflow = true[Ellipsis, i,j] - pred[Ellipsis, i,j]
                arr = quantile * tf.maximum(prediction_underflow, 0.) + (1. - quantile) * tf.maximum(-prediction_underflow, 0.)
                loss = loss + tf.reduce_sum(input_tensor=arr, axis=-1)


        return tf.reduce_mean(loss)


    def Train(self, X_train, Y_train, sArtifactsFolder, fLearningRate, iNrOfEpochs, iBatchSize ,oLoss, oMetrics):
        self.compile(
            loss = oLoss,
            metrics = oMetrics,
            optimizer= Adam(
                learning_rate=ExponentialDecay(
                    initial_learning_rate=fLearningRate,
                    decay_steps=10**2,
                    decay_rate=0.9
                )
            )
        )

        self.fit(
            X_train,
            Y_train,
            epochs= iNrOfEpochs,
            batch_size=iBatchSize,
            verbose=1
        )


        sModelArtifactPath = f'{sArtifactsFolder}\\'
        self.save_weights(
            sModelArtifactPath,
            save_format ='tf'
        )