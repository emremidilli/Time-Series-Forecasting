'''
    In original MultiHeadAttention layer of Keras, for each head a different
        values are used.
    This makes interpretability difficult.
    That's why the authors of TFT proposed InterpretableMultiHeadAttention
        where values are shared accross heads.
'''

import tensorflow as tf


def get_decoder_mask(self_attn_inputs):
    '''
    Returns causal mask to apply for self-attention layer.
    Args:
        self_attn_inputs: Inputs to self attention layer to determine mask
            shape
    '''
    len_s = tf.shape(input=self_attn_inputs)[1]
    bs = tf.shape(input=self_attn_inputs)[:1]
    mask = tf.keras.backend.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class ScaledDotProductAttention():
    '''
    Defines scaled dot product attention layer.
    Attributes:
        dropout: Dropout rate to use
        activation: Normalisation function for scaled dot product attention
            (e.g.softmax by default)
    '''
    def __init__(self, attn_dropout=0.0):
        self.dropout = tf.keras.layers.Dropout(attn_dropout)
        self.activation = tf.keras.layers.Activation('softmax')

    def __call__(self, q, k, v, mask):
        """
            Applies scaled dot product attention.
                Args:
                  q: Queries
                  k: Keys
                  v: Values
                  mask: Masking if required -- sets softmax to very large value

            Returns:
              Tuple of (layer outputs, attention weights)
          """
        temper = tf.sqrt(tf.cast(tf.shape(input=k)[-1], dtype='float32'))
        attn = tf.keras.layers.Lambda(lambda x: tf.keras.backend.batch_dot(
            x[0], x[1], axes=[2, 2]) / temper)([q, k])  # shape=(batch, q, k)
        if mask is not None:
            mmask = tf.keras.layers.Lambda(
                lambda x: (-1e+9) * (1. - tf.keras.backend.cast(x, 'float32')))(mask)  # setting to infinity
            attn = tf.keras.layers.Add()([attn, mmask])

        attn = self.activation(attn)
        attn = self.dropout(attn)
        output =  tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class interpretable_multi_head_attention(tf.keras.layers.Layer):

    def __init__(self,iNrOfHeads, iModelDims, fDropout, **kwargs):
        super().__init__(**kwargs)

        self.iNrOfHeads = iNrOfHeads
        self.iModelDims = iModelDims

        self.iKeyDims = self.iModelDims // self.iNrOfHeads
        self.iValueDims = self.iModelDims // self.iNrOfHeads


        self.aQueries = []
        self.aKeys = []
        self.aValues = []
        oSharedValue = tf.keras.layers.Dense(units = self.iValueDims, use_bias = False, activation = None)
        for i in range(self.iNrOfHeads):
            self.aQueries.append(
                tf.keras.layers.Dense(units = self.iKeyDims, use_bias = False, activation = None)
            )

            self.aKeys.append(
                tf.keras.layers.Dense(units = self.iKeyDims, use_bias = False, activation = None)
            )

            self.aValues.append(oSharedValue)


        self.oAttention = ScaledDotProductAttention()
        self.oDropout = tf.keras.layers.Dropout(rate = fDropout)
        self.oDense = tf.keras.layers.Dense(units = iModelDims, use_bias = False)



    def call(self, q, k, v, mask = None):


        aHeads = []
        aAttentions = []

        for i in range(self.iNrOfHeads):
            q_i = self.aQueries[i](q)
            k_i = self.aKeys[i](k)
            v_i = self.aValues[i](v)

            head , attn = self.oAttention(
                q_i,
                k_i,
                v_i,
                mask
            )
            head_dropped = self.oDropout(head)

            aHeads.append(head)
            aAttentions.append(attn)


        if self.iNrOfHeads > 1:
            head = tf.stack(aHeads)
        else:
            head = aHeads[0]

        attn = tf.stack(aAttentions)

        if self.iNrOfHeads > 1:
            y = tf.keras.backend.mean(head, axis=0)
        else:
            y = head

        y = self.oDense(y)
        y = self.oDropout(y)

        return y, attn
