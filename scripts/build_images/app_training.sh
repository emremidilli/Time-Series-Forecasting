#!/bin/bash

cd ../../app_training/

docker-compose down

docker-compose build
