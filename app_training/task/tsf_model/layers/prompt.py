import tensorflow as tf

@tf.keras.saving.register_keras_serializable()
class SoftPrompts(tf.keras.layers.Layer):
    '''Soft prompts'''
    def __init__(
        self,
        patch_size,
        embedding_dims,
        temporal_dims,
        pool_size,
        nr_of_most_similar_prompts,
        **kwargs
        ):
        '''
        A prompt value is in shape of (temporal_dims, embedding_dims).
        A prompt pool is a list key-value pairs in which
            the value has shape of (pool_size, temporal_dims, embedding_dims).
        Prompt value is a trainable weight.
        A prompt key is a vector that has shape of
            (pool_size, embedding_dims).

        args:
            patch_size (int) - patch size.
            embedding_dims (int) -  embedding dimension.
            temporal_dims (int) - temporal dimension of prompt.
            pool_size (int) - number of the prompts in the pool.
            nr_of_most_similar_prompts (int) - number of the most similar
                prompts to be returned.
        '''

        super().__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.patch_size = patch_size
        self.embedding_dims = embedding_dims
        self.temporal_dims = temporal_dims
        self.pool_size = pool_size
        self.nr_of_most_similar_prompts = nr_of_most_similar_prompts

        self.prompt_keys = tf.random.uniform(shape=(pool_size, embedding_dims))

        self.prompt_values = tf.random.uniform(shape=(pool_size, embedding_dims))
