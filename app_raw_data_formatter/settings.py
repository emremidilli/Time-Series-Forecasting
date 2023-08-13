SERVER = '' #/home/yunusemre
BIN_FOLDER = f'{SERVER}/TSF-bin'

RAW_DATA_FOLDER = f'{BIN_FOLDER}/00 - Raw Data'
CONVERTED_DATA_FOLDER = f'{BIN_FOLDER}/01 - Converted Data'

RAW_FREQUENCY  = 'T' # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
RAW_FREQUENCY_NUMPY = 'm' # https://numpy.org/doc/stable/reference/arrays.datetime.html

PATCH_SIZE= 30
FORECAST_HORIZON = 120
LOOKBACK_COEFFICIENT = 4
TEST_SIZE = 500
