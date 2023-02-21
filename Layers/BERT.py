import tensorflow as tf
from PositionEncoder import PositionEncoder 
from TransformerEncoder import TransformerEncoder 

    
def get_bert_model(
    mlm_input_shape, 
    nsp_input_shape,
    nr_of_encoder_blocks,
    attention_key_dims,
    attention_nr_of_heads,
    attention_dense_dims,
    dropout_rate    
):
    mlm_input = tf.keras.layers.Input(mlm_input_shape, name = 'mlm_input')
    nsp_input = tf.keras.layers.Input(nsp_input_shape, name = 'nsp_input')

    x = tf.keras.layers.Concatenate(axis = 1, name = 'concatenate_inputs')([mlm_input, nsp_input])
    x = PositionEncoder(
        input_dim = mlm_input_shape[0] + nsp_input_shape[0] , 
        output_dim = mlm_input_shape[1]
    )(x)
    
    
    for i in range(nr_of_encoder_blocks):
        x = TransformerEncoder(
                attention_key_dims, 
                attention_nr_of_heads, 
                attention_dense_dims,
                dropout_rate,
                mlm_input_shape[-1]
            )(x)

    mlm_output = tf.keras.layers.Dense(mlm_input_shape[1], activation = 'softmax', name = 'mlm_classifier')(x)

    x = tf.keras.layers.Flatten()(x)
    nsp_output = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'nsp_classifier')(x)

    oModel = tf.keras.Model(inputs = [mlm_input, nsp_input], outputs= [mlm_output, nsp_output])

    return oModel



def mlm_custom_loss( y_true, y_pred):
    # y_true for mlm is without cls token. So we should ignore the cls_token time step of y_pred.
    # since cls token is the latest latest token of prediction, we ignore that token.
    y_pred_without_special_tokens = y_pred[:, :-1, :]
    non_masks = tf.not_equal(y_true, tf.cast(tf.ones(y_true.shape[2])*-1, tf.dtypes.float32))[:,:,0]

    y_pred_non_mask = tf.boolean_mask(y_pred_without_special_tokens,non_masks)
    y_true_non_mask = tf.boolean_mask(y_true,non_masks)

    oBinCE = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    loss = oBinCE(y_true_non_mask , y_pred_non_mask)
    loss = tf.reduce_mean(loss)

    return loss

def nsp_custom_loss( y_true, y_pred):
    oBinCE = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    loss = oBinCE(y_true , y_pred)
    loss = tf.reduce_mean(loss)

    return  loss

