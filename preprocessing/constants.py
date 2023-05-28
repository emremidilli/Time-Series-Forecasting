# folder paths
SCALERS_FOLDER = r'C:\Users\yunus\Desktop\Scalers'
RAW_DATA_FOLDER = r'C:\Users\yunus\Desktop\Raw Data'
CONVERTED_DATA_FOLDER = r'C:\Users\yunus\Desktop\Converted Data'
SCALED_DATA_FOLDER = r'C:\Users\yunus\Desktop\Scaled Data'
CONSOLIDATED_CHANNEL_DATA_FOLDER = r'C:\Users\yunus\Desktop\Consolidated Channel Data'
TOKENIZIED_DATA_FOLDER = r'C:\Users\yunus\Desktop\Tokenized Data'
NEXT_PATCH_PREDICTION_DATA_FOLDER = r'C:\Users\yunus\Desktop\Next Patch Prediction Data'
MASKED_PATCH_PREDICTION_DATA_FOLDER = r'C:\Users\yunus\Desktop\Masked Patch Prediction Data'
SIGN_OF_PATCH_PREDICTION_DATA_FOLDER = r'C:\Users\yunus\Desktop\Sign of Patch Prediction Data'
RANK_OF_PATCH_PREDICTION_DATA_FOLDER = r'C:\Users\yunus\Desktop\Rank of Patch Prediction Data'
DATE_OF_PATCH_PREDICTION_DATA_FOLDER =r'C:\Users\yunus\Desktop\Date of Patch Prediction Data'

THRESHOLD_STATIC_SENSITIVITY = 0.01

RAW_FREQUENCY  = 'T' # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
DATETIME_FEATURES = ['month', 'day', 'dayofweek', 'hour', 'minute']

PATCH_SIZE= 30
FORECAST_HORIZON = 120
LOOKBACK_COEFFICIENT = 4

NR_OF_BINS = 8

PATCH_SAMPLE_RATE = 0.15
POOL_SIZE = 3

CLS_SCALAR = 0.51
CNL_SCALAR = 0.52
MSK_SCALAR = 0.53
SEP_SCALAR = 0.54

MASK_RATE = 0.70
FALSE_NEXT_PATCH_RATE = 0.50