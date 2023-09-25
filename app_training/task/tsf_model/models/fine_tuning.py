import tensorflow as tf

from tsf_model.layers.decoder import SingleStepDecoder


@tf.keras.saving.register_keras_serializable()
class FineTuning(tf.keras.Model):
    '''Keras model for fine-tuning purpose.'''
    def __init__(self,
                 con_temp_pret,
                 nr_of_time_steps,
                 **kwargs):
        super().__init__(**kwargs)

        self.con_temp_pret = con_temp_pret
        self.con_temp_pret.trainable = False

        self.single_step_decoder = SingleStepDecoder(
            nr_of_time_steps=nr_of_time_steps)

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
            Timesteps of forecast horizon are masked.
        output: (none, timesteps, 1)
        '''
        y_cont_temp = self.con_temp_pret(inputs)

        y_pred = self.single_step_decoder(y_cont_temp)
        return y_pred
