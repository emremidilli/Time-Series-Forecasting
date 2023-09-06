import tensorflow as tf


def get_pre_trained_representation(pre_trained_model_dir):
    pre_trained_model = tf.keras.models.load_model(pre_trained_model_dir)

    con_temp_pret = pre_trained_model.encoder_representation

    return con_temp_pret
