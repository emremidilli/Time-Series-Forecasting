import tensorflow as tf


class GatedLinearUnit(tf.keras.layers.Layer):
    '''
    Gated linear unit from "Temporal Fusion Transformer" paper
    It suppress the input that is not relevant for the given task.
    '''
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.linear = tf.keras.layers.Dense(units)
        self.sigmoid = tf.keras.layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)


class GatedResidualNetwork(tf.keras.layers.Layer):
    '''
    Gated residual network from "Temporal Fusion Transformer"
    The Gated Residual Network (GRN) works as follows:
        1. Applies the nonlinear ELU transformation to the inputs.
        2. Applies linear transformation followed by dropout.
        3. Applies GLU and adds the original inputs to the output of the GLU
            to perform skip (residual) connection.
        4. Applies layer normalization and produces the output.
    '''
    def __init__(self, units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.elu_dense = tf.keras.layers.Dense(units, activation="elu")
        self.linear_dense = tf.keras.layers.Dense(units)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.project = tf.keras.layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x


class VariableSelection(tf.keras.layers.Layer):
    '''
    Variable selection network of "Temporal Fusion Transformer"
    The Variable Selection Network (VSN) works as follows:
        1. Applies a GRN to each feature individually.
        2. Applies a GRN on the concatenation of all the features,
            followed by a softmax to produce feature weights.
        3. Produces a weighted sum of the output of the individual GRN.
    Note that the output of the VSN is [batch_size, encoding_size],
        regardless of the number of the input features.
    '''
    def __init__(self, num_features, units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = tf.keras.layers.Dense(units=num_features,
                                             activation="softmax")

    def call(self, inputs):
        '''
        inputs: tuple of inputs each has same shape of (1, dims)
        '''
        v = tf.keras.layers.concatenate(inputs)
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        x = tf.stack(x, axis=1)

        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs
