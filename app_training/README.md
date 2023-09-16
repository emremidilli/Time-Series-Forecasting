# Objective

Trains given datasets. There are two training jobs:

1. Pre-training

2. Fine-tuning

# Inputs

Inputs are in tf.data.Dataset format. Input dataset contains 4 compoenents:

1. Distribution

2. Trend

3. Seasonality

4. Datetime features

# Targets

Targets are in tf.data.Dataset format. Target dataset contains 1 component:

1. Quantiles