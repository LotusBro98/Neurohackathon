import tensorflow as tf
from tensorflow import keras

import os
import pickle

dataset = pickle.load(open("datasets/data.pickle", "rb"))
dataset2 = pickle.load(open("datasets/data.pickle", "rb"))

features = dataset['features']
labels = dataset['labels']
Nhistory = dataset['Nhistory']

features2 = dataset['features']
labels2 = dataset['labels']

train_features = features
train_labels = labels

test_features = features2
test_labels = labels2

print(labels)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(Nhistory, 6)),
    #keras.layers.Dense(6, activation=tf.nn.tanh),
    #keras.layers.Dense(200, activation=tf.nn.tanh),
    keras.layers.Dense(100, activation=tf.nn.tanh),
    keras.layers.Dense(50, activation=tf.nn.tanh),
    #keras.layers.Dense(10, activation=tf.nn.tanh),
    #keras.layers.Dense(5, activation=tf.nn.sigmoid),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_path = "net/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

if (os.path.isfile("net/checkpoint")):
    print("Loading weights from file")
    model.load_weights(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model.fit(
    train_features, train_labels, epochs=100000,
    validation_data= (test_features, test_labels),
    callbacks=[cp_callback],
    batch_size=20,
)

#~~~~~~~~~~~~~~~~~

test_loss, test_accuarcy = model.evaluate(train_features, train_labels)

print("Test accuracy", str(int(test_accuarcy * 100)) + "%")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#print(model.summary())