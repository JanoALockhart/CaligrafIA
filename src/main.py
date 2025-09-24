import settings

import tensorflow as tf
import keras
from keras import layers
from dataloader import IAMLineDataloader

input_shape = (32, 256, 1)
alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

# DATASETS


# MODEL
model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=5, padding="same", input_shape=input_shape),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
    
    layers.Conv2D(filters=64, kernel_size=5, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
    
    layers.Conv2D(filters=128, kernel_size=3, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding="valid"),
    
    layers.Conv2D(filters=128, kernel_size=3, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding="valid"),
    
    layers.Conv2D(filters=256, kernel_size=3, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding="valid"),

    layers.Reshape((64, 256)),

    layers.Bidirectional(layers.LSTM(256, return_sequences=True), merge_mode="concat"),
    layers.Bidirectional(layers.LSTM(256, return_sequences=True), merge_mode="concat"),

    layers.Dense(len(alphabet) + 1, activation=None),
])

model.summary()

# COMPILE
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CTC(),
)

# TRAINING