#!/bin/bash

app_name=$1
cd ../$app_name/

docker-compose build

IMAGE_NAME="$app_name:latest"

REPO_NAME=senaacers55/tsf-pre-training
TAG_NAME=$app_name
IMAGE_URI=$REPO_NAME:$TAG_NAME

docker tag $IMAGE_NAME $IMAGE_URI

docker login
docker push $IMAGE_URI