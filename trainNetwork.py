import tensorflow as tf
from tensorflow import keras

import os
import pickle

# fashion_mnist = keras.datasets.fashion_mnist
#
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# train_images = train_images / 255.0
# test_images = test_images / 255.0

dataset = pickle.load(open("datasets/data.pickle", "rb"))

features = dataset['features']
labels = dataset['labels']
Nhistory = dataset['Nhistory']

train_features = features[:5000]
train_labels = labels[:5000]

test_features = features
test_labels = labels

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(Nhistory, 6)),
    #keras.layers.Dense(6, activation=tf.nn.tanh),
    #keras.layers.Dense(20, activation=tf.nn.tanh),
    #keras.layers.Dense(10, activation=tf.nn.tanh),
    keras.layers.Dense(5, activation=tf.nn.sigmoid),
    #keras.layers.Dense(5, activation=tf.nn.sigmoid)
])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_path = "net/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

if (os.path.isfile(checkpoint_path)):
    model.load_weights(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model.fit(
    train_features, train_labels, epochs=7,
    validation_data= (train_features, train_labels),
    callbacks=[cp_callback],
    batch_size=20
)

#~~~~~~~~~~~~~~~~~

test_loss, test_accuarcy = model.evaluate(test_features, test_labels)

print("Test accuracy", str(int(test_accuarcy * 100)) + "%")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(model.summary())