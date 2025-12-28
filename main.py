import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# print(tf.__version__)

# Initialization of Tensor
# y = tf.eye(3)
# x = tf.random.normal((3, 3), mean=0, stddev=1)
# print(x)

#Mathematical Operations
# x = tf.constant([1, 2, 3])
# y = tf.constant([9, 8, 7])
#
# z = tf.add(x, y)
# k = tf.subtract(x, y)
# q = tf.tensordot(x, y, axes=1)
# print(z)
# print(k)
# print(q)

# x = tf.random.normal((2,3))
# y = tf.random.normal((3,4))
#
# z = tf.matmul(x,y)
# print(z)

#Indexing of Tensor

#Loading dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

# Sequential API
model = keras.Sequential([
    keras.Input(shape=(28,28)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10),
])

#print(model.summary()) #This is a common debugging tool.

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'],
)

#print(model.summary())

# model.fit(x_train, y_train, batch_size = 32, epochs = 5, verbose = 2)
# model.evaluate(x_test, y_test, batch_size = 32, verbose = 2)

sys.exit()

#Functional API
inputs = keras.Input(shape=(28*28,))
x = layers.Dense(64, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x)
models = keras.Model(inputs=inputs, outputs=outputs)

print(models.summary())

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['accuracy'],
)

