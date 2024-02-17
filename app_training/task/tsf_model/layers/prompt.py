import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class SoftPrompts(tf.keras.layers.Layer):
    '''Soft prompts with prompt pooling.'''
    def __init__(
            self,
            key_dims,
            embedding_dims,
            prompt_length,
            pool_size,
            nr_of_most_similar_prompts,
            **kwargs):
        '''
        A prompt pool is a list key-value pairs.
        Prompt value is a trainable weight with the shape of
            (prompt_length, embedding_dimss)
        A prompt key is a vector that has shape of
            (1, key_dims).

        args:
            key_dims (int) - key dimension.
            embedding_dims (int) -  embedding dimension.
            prompt_length (int) - prompt length.
            pool_size (int) - number of the prompts in the pool.
            nr_of_most_similar_prompts (int) - number of the most similar
                prompts to be returned.
        '''

        super().__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.prompt_length = prompt_length
        self.pool_size = pool_size
        self.nr_of_most_similar_prompts = nr_of_most_similar_prompts
        self.key_dims = key_dims

        self.prompt_keys = tf.random.uniform(shape=(pool_size, key_dims))

        self.prompt_values = tf.random.uniform(
            shape=(pool_size, prompt_length, embedding_dims))

    def call(self, inputs):
        '''
        Calculates the most similar prompts by keys.
        Returns the values of them.

        inputs: (None, timesteps)
        returns: (None, timesteps, features)
        '''
        # Reshape inputs to match the prompt keys' shape
        inputs = tf.expand_dims(inputs, axis=1)

        # Compute cosine similarity between inputs and prompt keys
        similarity_scores = tf.keras.losses.cosine_similarity(
            inputs,
            self.prompt_keys,
            axis=-1
        )
        # Get indices of most similar prompts
        top_indices = tf.math.top_k(
            similarity_scores,
            k=self.nr_of_most_similar_prompts).indices

        # Gather the values of the most similar prompts
        most_similar_prompt_values = tf.gather(self.prompt_values, top_indices)

        return most_similar_prompt_values
