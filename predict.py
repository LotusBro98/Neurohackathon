import tensorflow as tf
from tensorflow import keras

import os
import pickle
import numpy as np

dataset = pickle.load(open("datasets/data2.pickle", "rb"))

features = dataset['features']
labels = dataset['labels']
Nhistory = dataset['Nhistory']

#index = int(input())
#index = index - firstSample - Nhistory + 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(Nhistory, 6)),
    keras.layers.Dense(6, activation=tf.nn.tanh),
    keras.layers.Dense(20, activation=tf.nn.tanh),
    #keras.layers.Dense(5, activation=tf.nn.sigmoid),
    keras.layers.Dense(5, activation=tf.nn.sigmoid)
])

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_path = "net/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

#~~~~~~~~~~~~~~~~~

#prediction = model.predict(np.asarray([sample_features]))[0]
#print(np.argmax(prediction))

