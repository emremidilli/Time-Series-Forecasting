import tensorflow as tf
print("Tensorflow Version:", tf.__version__)


import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#### Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images[0:160]
train_labels = train_labels[0:160]
test_images = test_images[0:160]
test_labels = test_labels[0:160]

train_images1 = train_images[:,:,:,np.newaxis]
test_images1 = test_images[:,:,:,np.newaxis]
##Scale these values to a range of 0 to 1 before feeding them to the neural network model
### Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

##Create the convolutional base
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

###Train the model
##Feed the model
history = model.fit(train_images1, train_labels, epochs=10,
                    validation_data=(test_images1, test_labels))