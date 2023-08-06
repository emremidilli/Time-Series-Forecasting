SERVER = ''

# folder paths
BIN_FOLDER = f'{SERVER}/TSF-bin'
TRAINING_DATA_FOLDER = f'{BIN_FOLDER}/02 - Training Datasets'
HYPERPARAMETER_TUNING_FOLDER = f'{BIN_FOLDER}/03 - Hyperparameter Tuning'
ARTIFACTS_FOLDER = f'{BIN_FOLDER}/04 - Artifacts'

# hyperparameter tuning
ARCHITECTURE_CONFIG = {
    'nr_of_encoder_blocks': [2, 6, 1],
    'nr_of_heads': [2, 16, 2],
    'nr_of_ffn_units_of_encoder': [8, 128, 16],
    'embedding_dims': [8, 128, 16],
    'dropout_rate': [0.01, 0.9, 0.1]}
OPTIMIZER_CONFIG = {
    'learning_rate': [0.0001, 0.01, 0.001],
    'momentum_rate': [0.1, 0.9, 0.1]}

# architectures
PROJECTION_HEAD = 32

# fine tuning parameters
DATETIME_FEATURES = ['month', 'day', 'dayofweek', 'hour', 'minute']
TARGET_QUANTILES = [0.10, 0.50, 0.90]

FORECAST_HORIZON = 120
LOOKBACK_COEFFICIENT = 4
PATCH_SIZE = 30
NR_OF_BINS = 8  # for DisERT
PATCH_SAMPLE_RATE = 0.15  # for TreERT & SeaERT
POOL_SIZE = 3  # for TreERT & SeaERT
MASK_RATE = 0.70
MSK_SCALAR = 0.53
PRE_TRAIN_RATIO = 0.20
MINI_BATCH_SIZE = 128
NR_OF_EPOCHS = 1000
PATIENCE = 20

NR_OF_LOOKBACK_PATCHES = int(
    (FORECAST_HORIZON * LOOKBACK_COEFFICIENT) / PATCH_SIZE)
NR_OF_FORECAST_PATCHES = int(FORECAST_HORIZON / PATCH_SIZE)
