import tensorflow as tf

from tsf_model.layers.ConTemPreT.decoder import QuantileDecoder


class FineTuning(tf.keras.Model):
    '''
        Keras model for fine-tuning purpose.
    '''
    def __init__(self,
                 con_tem_pret_model,
                 nr_of_quantiles,
                 **kwargs):
        super().__init__(**kwargs)

        self.con_tem_pret = con_tem_pret_model

        self.quantile_decoder = QuantileDecoder(nr_of_quantiles)

    def call(self, inputs):
        '''
            input: tuple of 4 arrays.
                1. dist: (none, timesteps, features)
                2. tre: (none, timesteps, features)
                3. sea: (none, timesteps, features)
                4. date: (none, features)
                Timesteps of forecast horizon are masked.
        '''
        y_cont_temp = self.con_tem_pret(inputs)

        y_pred = y_cont_temp
        return y_pred
