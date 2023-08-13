SERVER = ''

# folder paths
BIN_FOLDER = f'{SERVER}/TSF-bin'
TRAINING_DATA_FOLDER = f'{BIN_FOLDER}/02 - Training Datasets'
HYPERPARAMETER_TUNING_FOLDER = f'{BIN_FOLDER}/03 - Hyperparameter Tuning'
ARTIFACTS_FOLDER = f'{BIN_FOLDER}/04 - Artifacts'

# default architecture hyperparameters
NR_OF_ENCODER_BLOCKS = 4
NR_OF_HEADS = 2
DROPOUT_RATE = 0.10
ENCODER_FFN_UNITS = 32
EMBEDDING_DIMS = 32
PROJECTION_HEAD = 32

# default optimizer hyperparameters
LEARNING_RATE = 1e-5
BETA_1 = 0.90
BETA_2 = 0.99

# fine-tuning parameters
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

NR_OF_LOOKBACK_PATCHES = int(
    (FORECAST_HORIZON * LOOKBACK_COEFFICIENT) / PATCH_SIZE)
NR_OF_FORECAST_PATCHES = int(FORECAST_HORIZON / PATCH_SIZE)
