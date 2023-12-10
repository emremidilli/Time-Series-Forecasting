# Folder Structure

# Objective

To convert numpy datasets to the tensorflow datasets.

# Input

Input of the pipeline can be only the numpy datasets in 3 formats:

1. input:
    1. lookback: (nr_of_samples, nr_of_covariates, lookback_horizon)
    2. timestamps: (nr_of_samples, feature_size)
2. label:
    1. forecast: (nr_of_samples, nr_of_covariates forecast_horizon)

Inputs must be in .npy format.

# Output

Outputs of the pipeline are as follows:

```
└── 03_preprocessing
    └── model_id
        ├── dataset
        ├── input_preprocessor
        └── target_preprocessor
```

1. dataset - tf.data.Dataset objects.
2. input_preprocessor - pipeline object.
3. target_preprocessor - pipeline object.