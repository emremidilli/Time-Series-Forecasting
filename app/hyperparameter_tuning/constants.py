PRE_TRAINING_CONFIG = {
    'optimizer' : {
        'learning_rate' : [0.0001,0.01, 0.001],
        'momentum_rate' : [0.1,0.9, 0.1]
    },
    'architecture' : {
        'nr_of_encoder_blocks' : [2, 6, 1],
        'nr_of_heads' : [2, 16, 2],
        'nr_of_ffn_units_of_encoder' : [16, 128, 16],
        'embedding_dims' : [4, 64, 6],
        'dropout_rate':[0.01,0.9, 0.1]   
    }
}


NR_OF_EPOCHS = 10
PATIENCE = 5
FACTOR = 3
SAMPLE_SIZE = 1000