FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="yunusemremidilli@gmail.com"

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

COPY ./requirements.txt /tmp/requirements.txt
COPY /app /app
COPY /scripts/pre_train.sh /scripts/pre_train.sh

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

WORKDIR /scripts

CMD ["sh", "pre_train.sh"]