import settings
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from dataloader import IAMLineDataloader

def main():
    input_shape = (32, 256, 1)

    # DATASETS
    # Create dataset for IAM
    iam_dataloader = IAMLineDataloader(settings.IAM_PATH)
    (samples, labels) = iam_dataloader.load_samples_tensor()

    # Add characters from dataset to alphabet
    unique_chars = set()
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

        return img, label

    # Splits
    total = len(samples)
    train_split = int(settings.TRAIN_SPLIT * total)
    val_split = int(settings.VAL_SPLIT * total)
    test_split = total - train_split - val_split

    train_samples = samples[0:train_split]
    train_labels = labels[0:train_split]
    train_ds = tf.data.Dataset.from_tensor_slices((train_samples, train_labels))
    train_ds = train_ds.map(preprocess_sample).padded_batch(settings.BATCH_SIZE, drop_remainder=True)

    val_samples = samples[train_split:train_split+val_split]
    val_labels = labels[train_split:train_split+val_split]
    val_ds = tf.data.Dataset.from_tensor_slices((val_samples, val_labels))
    val_ds = val_ds.map(preprocess_sample).padded_batch(settings.BATCH_SIZE)

    test_samples = samples[train_split+val_split:]
    test_labels = labels[train_split+val_split:]
    test_ds = tf.data.Dataset.from_tensor_slices((test_samples, test_labels))
    test_ds = test_ds.map(preprocess_sample).padded_batch(settings.BATCH_SIZE)

    print("Splits:  ",len(train_samples), len(val_samples), len(test_samples), len(samples))
    print("Batched: ",len(train_ds), len(val_ds), len(test_ds))

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
        layers.Dense(len(unique_chars) + 1, activation=None),
    ])

    model.summary()

    # COMPILE
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CTC())

    # TRAINING
    history = model.fit(x=train_ds, epochs=3, validation_data=val_ds)
    print(history.history)

if __name__ == "__main__":
    main()