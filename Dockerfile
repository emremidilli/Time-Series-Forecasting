FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="yunusemremidilli@gmail.com"

ENV PYTHONUNBUFFERED 1

COPY /app /app

WORKDIR /app/training

CMD ["python", "./pre_train.py"]