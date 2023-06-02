import sys
sys.path.append( '../')
from layers.temporal_fusion_transformer.variable_selection_network import variable_selection_network
from layers.temporal_fusion_transformer.interpretable_multi_head_attention import interpretable_multi_head_attention
from layers.temporal_fusion_transformer.gated_residual_network import gated_residual_network
from layers.temporal_fusion_transformer.gated_linear_unit import gated_linear_unit


import tensorflow as tf

class temporal_fusion_transformer(tf.keras.Model):
    
    def __init__(self, 
                 iNrOfLookbackPatches,
                 iNrOfForecastPatches,
                 **kwargs):
        super().__init__(**kwargs)
        
        iNrOfChannels = 3
        fDropout = 0.1
        iModelDims = 32
        
        
        self.oStaticEncoder = tf.keras.layers.Dense(units = iModelDims)
        
        self.oVsnStatic = variable_selection_network(
            iModelDims = iModelDims,
            iNrOfVariables = 2 * iNrOfChannels, # one for static digit, one for transition.`
            fDropout = fDropout,
            bIsWithExternal = False
        )
        
        self.oStaticLookbackRepeat = tf.keras.layers.RepeatVector(n = iNrOfLookbackPatches)
        self.oStaticForecastRepeat = tf.keras.layers.RepeatVector(n = iNrOfForecastPatches)
        self.oStaticAllRepeat = tf.keras.layers.RepeatVector(n = iNrOfLookbackPatches+ iNrOfForecastPatches)
    
        # static covariate encoders to be used in different parts of the tft model
        self.oStaticContextTemporalVsn = gated_residual_network(
            iInputDims = iModelDims ,
            iOutputDims = iModelDims, 
            fDropout = fDropout, 
            bIsWithStaticCovariate=False
        )
        
#         self.oStaticContextStateH = gated_residual_network(
#             iInputDims = iModelDims ,
#             iOutputDims = iModelDims, 
#             fDropout = fDropout, 
#             bIsWithStaticCovariate=False
#         )

#         self.oStaticContextStateC = gated_residual_network(
#             iInputDims = iModelDims ,
#             iOutputDims = iModelDims, 
#             fDropout = fDropout, 
#             bIsWithStaticCovariate=False
#         )
        
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
        
        
#         self.oInterpretableMha = interpretable_multi_head_attention(
#             iNrOfHeads = 2,
#             iModelDims = iModelDims,
#             fDropout = fDropout
#         )
        
        
    def call(self, x_lookback, x_forecast , x_static):
        
        s = self.oStaticEncoder(x_static)
        s_c, s_v = self.oVsnStatic(s)
        
        # producing static vectors via static covariate encoders
        s_c_temporal_vsn = self.oStaticContextTemporalVsn(s_c)
        s_c_static_enrichment = self.oStaticContextEnrichment(s_c)
        

        
        # variable selection for temporal steps
        s_c_lookback_vsn = self.oStaticLookbackRepeat(s_c_temporal_vsn)
        s_c_forecast_vsn = self.oStaticForecastRepeat(s_c_temporal_vsn)
        
        y_lookback, w_lookback = self.oTimeDistVsnLookback([x_lookback, s_c_lookback_vsn])
        y_forecast, w_forecast = self.oTimeDistVsnForecast([x_forecast, s_c_forecast_vsn])
        
        # encode & decode with LSTMs
        y_encoder, h_encoder, c_encoder = self.oLstmEncoder(y_lookback)        
        y_decoder, h_decoder, c_decoder = self.oLstmDecoder(y_forecast)
        
        # concat outputs
        y_lookback_forecast = self.oConcatter([y_lookback, y_forecast])
        y_encoder_decoder = self.oConcatter([y_encoder, y_decoder])
        
        # gate, add & norm
        y_tft_encoder = self.oGates_1(y_encoder_decoder)
        y_tft_encoder = y_tft_encoder + y_lookback_forecast 
        y_tft_encoder = self.oLayerNorm_1(y_tft_encoder)
        
        
        # static enrichment
        s_c_static_enrichment = self.oStaticAllRepeat(s_c_static_enrichment)
        y_static_enrichment = self.oStaticEnrichment(
            y_tft_encoder, s_c_static_enrichment
        )
        
        return y_static_enrichment
        
        