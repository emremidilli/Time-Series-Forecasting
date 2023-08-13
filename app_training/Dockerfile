FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-12.py310

LABEL maintainer="yunusemremidilli@gmail.com"

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

COPY /task /app
COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

WORKDIR /app

ENTRYPOINT ["python", "pre_train.py"]