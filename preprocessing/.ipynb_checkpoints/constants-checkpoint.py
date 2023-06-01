# folder paths
RAW_DATA_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\00 - Raw Data'
CONVERTED_DATA_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\01 - Converted Data'
SCALERS_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\02 - Scalers'
SCALED_DATA_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\02 - Scaled Data'
CONSOLIDATED_CHANNEL_DATA_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\03 - Consolidated Channel Data'
TOKENIZIED_DATA_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\04 - Tokenized Data'
NEXT_PATCH_PREDICTION_DATA_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\05 - Next Patch Prediction Data'
MASKED_PATCH_PREDICTION_DATA_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\06 - Masked Patch Prediction Data'
SIGN_OF_PATCH_PREDICTION_DATA_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\07 - Sign of Patch Prediction Data'
RANK_OF_PATCH_PREDICTION_DATA_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\08 - Rank of Patch Prediction Data'
DATE_OF_PATCH_PREDICTION_DATA_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\09 - Date of Patch Prediction Data'
QUANTILE_PREDICTION_DATA_FOLDER = r'C:\Users\yunus\Desktop\TSF-bin\10 - Quantile Prediction Data'

THRESHOLD_STATIC_SENSITIVITY = 0.01

RAW_FREQUENCY  = 'T' # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
DATETIME_FEATURES = ['month', 'day', 'dayofweek', 'hour', 'minute']


PATCH_SIZE= 30
FORECAST_HORIZON = 120
LOOKBACK_COEFFICIENT = 4

NR_OF_BINS = 8 #for DisERT

PATCH_SAMPLE_RATE = 0.15 # for TreERT & SeaERT
POOL_SIZE = 3 # for TreERT & SeaERT

MASK_RATE = 0.70
FALSE_NEXT_PATCH_RATE = 0.50

CLS_SCALAR = 0.51
CNL_SCALAR = 0.52
MSK_SCALAR = 0.53
SEP_SCALAR = 0.54