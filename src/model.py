
import keras
from keras import layers

from metrics import CharacterErrorRate, WordErrorRate
import settings

def build_model(input_shape, alphabet_length, int_to_char):
    model = keras.Sequential([
        keras.Input(shape=(input_shape)),
        layers.Conv2D(filters=32, kernel_size=5, padding="same"),
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
        layers.Dense(alphabet_length, activation=None),
    ])

    return model

    