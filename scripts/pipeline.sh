#!/bin/bash

channel=ETTh1

bash input_pipeline.sh

bash pre_train.sh $channel N 200

bash fine_tune.sh $channel N 250 Y 0.25

bash inference.sh $channel