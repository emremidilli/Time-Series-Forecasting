SERVER = ''

# folder paths
BIN_FOLDER = f'{SERVER}/tsf-bin'
TRAINING_DATA_FOLDER = f'{BIN_FOLDER}/02 - Training Datasets'
HYPERPARAMETER_TUNING_FOLDER = f'{BIN_FOLDER}/03 - Hyperparameter Tuning'
ARTIFACTS_FOLDER = f'{BIN_FOLDER}/04 - Artifacts'

# default architecture hyperparameters
NR_OF_ENCODER_BLOCKS = 4
NR_OF_HEADS = 4
DROPOUT_RATE = 0.10
ENCODER_FFN_UNITS = 128
EMBEDDING_DIMS = 128
PROJECTION_HEAD = 128

# default optimizer hyperparameters
LEARNING_RATE = 1e-5
BETA_1 = 0.90
BETA_2 = 0.99
CLIP_NORM = 1.0

# fine-tuning features
DATETIME_FEATURES = ['month', 'day', 'dayofweek', 'hour', 'minute']
TARGET_QUANTILES = [0.10, 0.50, 0.90]

# input representation
FORECAST_HORIZON = 120
LOOKBACK_COEFFICIENT = 4
PATCH_SIZE = 30
NR_OF_BINS = 8
POOL_SIZE_REDUCTION = 5
POOL_SIZE_TREND = 2
PRE_TRAIN_RATIO = 0.85
NR_OF_LOOKBACK_PATCHES = int(
    (FORECAST_HORIZON * LOOKBACK_COEFFICIENT) / PATCH_SIZE)
NR_OF_FORECAST_PATCHES = int(FORECAST_HORIZON / PATCH_SIZE)

# training
MASK_RATE = 0.70
MSK_SCALAR = 0.53
MINI_BATCH_SIZE = 128
NR_OF_EPOCHS = 10000
