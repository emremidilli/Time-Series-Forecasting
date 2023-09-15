# Objective

To convert numpy datasets to the tensorflow datasets.

# Input

Input of the pipeline can be only the numpy datasets in 3 formats:

1. input:
    1. lookback: (nr_of_samples, lookback_horizon)
    2. timestamps: (nr_of_samples, feature_size)
2. label:
    1. forecast: (nr_of_samples, forecast_horizon)

Inputs must be in .npy format.

# Output

Output of the pipeline are:
1. datasets - tf.data.Dataset objects.
2. pipeline - pipeline object.
