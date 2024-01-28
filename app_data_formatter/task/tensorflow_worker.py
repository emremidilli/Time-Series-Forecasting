# tensorflow_worker.py
from celery import Celery

app = Celery(
    'tensorflow_worker',
    broker='')


@app.task
def process_tensorflow_task(message):
    # Extract the necessary information from the m
    args = message['args']
    kwargs = message['kwargs']

    print(f"""Hello !
            I've received your message.
            Don't worry \n args: {args} \n kwargs:{kwargs}""")

    # Perform TensorFlow-related logic here using args and kwargs
    # ...

# Start the worker
# celery -A tensorflow_worker worker --loglevel=info
