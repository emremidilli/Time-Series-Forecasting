import tensorflow as tf
import numpy as np

class PositionEncoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim ):
        super(PositionEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.pos_enc = self.positional_encoding(input_dim, output_dim)
        
        self.dense = tf.keras.layers.Dense(
            self.output_dim, 
            name="position_embedding"
        )
        
    def call(self, x):
        
        y = self.dense(x)
    
        y = y + self.pos_enc
        return y

    
#     def get_config(self):
#         return {"input_dim": self.input_dim, "output_dim": self.output_dim}

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
    
    
    def positional_encoding(self,  length, depth):

        depth_to_return = depth

        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)

        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)

        pos_encoding = np.concatenate(
          [np.sin(angle_rads), np.cos(angle_rads)],
          axis=-1) 

        pos_encoding = pos_encoding[:, :depth_to_return]

        return tf.cast(pos_encoding, dtype=tf.float32)
    # def get_pos_encoding_matrix(self, max_len,d_emb ):
    #     pos_enc = np.array(
    #         [
    #             [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
    #             if pos != 0
    #             else np.zeros(d_emb)
    #             for pos in range(max_len)
    #         ]
    #     )
    #     pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    #     pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    #     return pos_enc

