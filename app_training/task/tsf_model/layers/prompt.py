import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class SoftPrompts(tf.keras.layers.Layer):
    '''Soft prompts with prompt pooling.'''
    def __init__(
            self,
            key_dims,
            embedding_dims,
            prompt_length,
            prompt_pool_size,
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
            prompt_pool_size (int) - number of the prompts in the pool.
            nr_of_most_similar_prompts (int) - number of the most similar
                prompts to be returned.
        '''

        super().__init__(**kwargs)
        self.embedding_dims = embedding_dims
        self.prompt_length = prompt_length
        self.prompt_pool_size = prompt_pool_size
        self.nr_of_most_similar_prompts = nr_of_most_similar_prompts
        self.key_dims = key_dims

        self.reshaper = tf.keras.layers.Reshape((key_dims,))

        # Initialize prompt keys and values using tf.Variable
        self.prompt_keys = self.add_weight(
            shape=(prompt_pool_size, key_dims),
            initializer='random_uniform',
            trainable=False,
            name='prompt_keys'
        )

        self.prompt_values = self.add_weight(
            shape=(prompt_pool_size, prompt_length, embedding_dims),
            initializer='random_uniform',
            trainable=False,
            name='prompt_values'
        )

    def call(self, inputs):
        '''
        Calculates the most similar prompts by keys.
        Returns the values of them.

        inputs: (None, timesteps, covariates)
        returns: (None, timesteps, features)
        '''
        # reduce multivariate to univariate
        # since expected input is already instance normalized,
        # there should not be problem with difference of
        # scales between the covariates.
        x = self.reshaper(inputs)

        # Reshape inputs to match the prompt keys' shape
        x = tf.expand_dims(x, axis=1)

        # Compute cosine similarity between inputs and prompt keys
        similarity_scores = tf.keras.losses.cosine_similarity(
            x,
            self.prompt_keys,
            axis=-1)

        # Get indices of most similar prompts
        top_indices = tf.math.top_k(
            similarity_scores,
            k=self.nr_of_most_similar_prompts).indices

        # Gather the values of the most similar prompts
        most_similar_prompt_values = \
            tf.gather(self.prompt_values, top_indices)

        # Reshape to required shape using tf.concat
        most_similar_prompt_values = \
            tf.concat(
                tf.unstack(most_similar_prompt_values, axis=1),
                axis=1)

        return most_similar_prompt_values
