import settings
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from dataloader import IAMLineDataloader

def main():
    input_shape = (32, 256, 1)
    alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

    # DATASETS
    # Create dataset for IAM
    iam_dataloader = IAMLineDataloader(settings.IAM_PATH)
    (samples, labels) = iam_dataloader.load_samples_tensor()
    dataset = tf.data.Dataset.from_tensor_slices((samples, labels))

    # Add characters from dataset to alphabet
    unique_chars = set(alphabet)
    max_length = 0
    longest_label = ""
    for label in labels:
        unique_chars.update(label)
        if max_length < len(label):
            max_length = len(label)
            longest_label = label
    unique_chars = sorted(unique_chars)
    
    # Create lambda char_to_int for CTC using StringLookup
    char_to_int = layers.StringLookup(vocabulary=unique_chars, oov_token="[UNK]")
    int_to_char = layers.StringLookup(vocabulary=unique_chars, oov_token="[UNK]", invert=True)

    print(unique_chars)
    print(max_length)
    print(longest_label)


    def preprocess_sample(img_path, label):
        """Opens image and convert image to tensor of floats. Process characters from label to int format for CTC"""
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize_with_pad(img, 32, 256)
        img = img / 255.0
        img = (img - 0.5) / 0.5

        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        label = char_to_int(label)
        label = tf.pad(label, [[0, max_length - tf.shape(label)[0]]])

        return img, label
    
    dataset = dataset.map(preprocess_sample).batch(32)

    # Splits
    total = len(dataset)
    train_split = int(0.95 * total)
    val_split = int(0.04 * total)
    test_split = total - train_split - val_split

    train_ds = dataset.take(train_split)
    val_ds = dataset.skip(train_split).take(val_split)
    test_ds = dataset.skip(train_split).skip(val_split)

    for (sample, label) in train_ds.take(1):
        print(sample.numpy().shape)
        print(label.numpy())

    # MODEL
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
        layers.Dense(len(alphabet) + 1, activation=None),
    ])

    model.summary()

    # COMPILE
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CTC())

    # TRAINING
    history = model.fit(x=train_ds, epochs=3, validation_data=val_ds)
    print(history.history)

if __name__ == "__main__":
    main()