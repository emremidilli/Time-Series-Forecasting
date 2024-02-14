# Objective

Trains given datasets. There are two training jobs:

1. Pre-training

2. Fine-tuning

# Pre-Training

It is a masked auto encoder training followed by contastive learning.

Pre-training is applied over a univariate dataset.

Pre-trained backbone is utilized in fine-tuning phase.

# Fine-Tuning

Fine-tuning model is multi-variate model. Each covariate shares the same backbone.

If there are N covariates, N linear heads are used.
