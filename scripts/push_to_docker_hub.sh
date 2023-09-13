#!/bin/bash

REPO_NAME=senaacers55/tsf-pre-training
TAG_NAME=time-series-forecasting-app_training
IMAGE_URI=$REPO_NAME:$TAG_NAME

docker login
docker tag $TAG_NAME:latest $IMAGE_URI
docker push $IMAGE_URI