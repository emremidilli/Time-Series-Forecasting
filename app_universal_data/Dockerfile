FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="yunusemremidilli@gmail.com"

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

COPY /task /app
COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

WORKDIR /app

ENTRYPOINT [ "python" ]
