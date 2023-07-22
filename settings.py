SERVER = '/home/yunusemre'
BIN_FOLDER = f'{SERVER}/TSF-bin'

# folder paths
RAW_DATA_FOLDER = f'{BIN_FOLDER}/00 - Raw Data'
CONVERTED_DATA_FOLDER = f'{BIN_FOLDER}/01 - Converted Data'
TRAINING_DATA_FOLDER = f'{BIN_FOLDER}/02 - Training Datasets'
HYPERPARAMETER_TUNING_FOLDER = f'{BIN_FOLDER}/10 - Hyperparameter Tuning'
ARTIFACTS_FOLDER = f'{BIN_FOLDER}/11 - Artifacts'

THRESHOLD_STATIC_SENSITIVITY = 0.01

RAW_FREQUENCY  = 'T' # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
RAW_FREQUENCY_NUMPY = 'm' # https://numpy.org/doc/stable/reference/arrays.datetime.html

DATETIME_FEATURES = ['month', 'day', 'dayofweek', 'hour', 'minute']
TARGET_QUANTILES = [0.10, 0.50, 0.90]

PATCH_SIZE= 30
FORECAST_HORIZON = 120
LOOKBACK_COEFFICIENT = 4

NR_OF_BINS = 8 #for DisERT

PATCH_SAMPLE_RATE = 0.15 # for TreERT & SeaERT
POOL_SIZE = 3 # for TreERT & SeaERT

MASK_RATE = 0.70
MSK_SCALAR = 0.53


TEST_SIZE = 500


BATCH_SIZE = 1000
MINI_BATCH_SIZE = 32
NR_OF_EPOCHS = 1000
TEST_SIZE = 500
PATIENCE = 20

NR_OF_LOOKBACK_PATCHES = int((FORECAST_HORIZON * LOOKBACK_COEFFICIENT)/PATCH_SIZE)
NR_OF_FORECAST_PATCHES = int(FORECAST_HORIZON/PATCH_SIZE)