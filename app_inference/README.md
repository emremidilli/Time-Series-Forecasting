# Objective

To produce prediction from inputs. Inputs are pre-processed based on the pre-processor models.

It picks the relevant prediction model.

# Input

Batch inputs are accepted in numpy format. There are two inputs that are necessary:

1. Lookback horizon and

2. Datetime features

# Output

Prediction of the given batch.