
import keras
from keras import layers

def build_model(input_shape, alphabet_length):
    input = keras.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=5, padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

    x = layers.Conv2D(filters=64, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
    
    
    x = layers.Conv2D(filters=128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding="valid")(x)
    
    x = layers.Conv2D(filters=128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding="valid")(x)
    
    x = layers.Conv2D(filters=256, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding="valid")(x)
    
    
    x = layers.Reshape((64, 256))(x)
    
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True), merge_mode="concat")(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True), merge_mode="concat")(x)

    output = layers.Dense(alphabet_length, activation=None)(x)

    return keras.Model(input, output)