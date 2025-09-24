import settings

import tensorflow as tf
import keras
from keras import layers
from dataloader import IAMLineDataloader

input_shape = (32, 256, 1)
alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

# DATASETS
def preprocess_sample(img_path, label):
    """Opens image and convert image to tensor of floats. Process characters from label to int format for CTC"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)

# Create dataset for IAM
iam_dataloader = IAMLineDataloader(settings.IAM_PATH)
(samples, labels) = iam_dataloader.load_samples_tensor()
dataset = tf.data.Dataset.from_tensor_slices((samples, labels))

# Splits
total = len(dataset)
train_split = int(0.95 * total)
val_split = int(0.04 * total)
test_split = total - train_split - val_split

train_ds = dataset.take(train_split)
val_ds = dataset.skip(train_split).take(val_split)
test_ds = dataset.skip(train_split).skip(val_split)




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