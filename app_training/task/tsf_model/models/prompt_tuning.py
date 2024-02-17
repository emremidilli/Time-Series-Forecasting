from layers import SoftPrompts

import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class PromptTuning(tf.keras.Model):
    '''Keras model for prompt-tuning purpose.'''
    def __init__(
            self,
            pool_length,
            prompt_length,
            embedding_dims,
            **kwargs):
        '''
        args:

        '''
        super().__init__(**kwargs)

        self.pool_length = pool_length
        self.prompt_length = prompt_length
        self.embedding_dims = embedding_dims

        self.soft_prompts = SoftPrompts()

    def call(self, inputs):
        '''
        inputs:

        '''

        # identify the most similar prompt from the pool.