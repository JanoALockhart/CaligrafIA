import json
import logging
from metrics import CharacterErrorRate, WordErrorRate
import settings
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
from keras import layers
from dataloader import IAMLineDataloader

from model import build_model
from callbacks import ValidationLogCallback, MetricsHistoryLogger

def main():
    input_shape = (32, 256, 1)

    logging.basicConfig(
        level=logging.INFO,
        filename=settings.VALIDATION_LOG_PATH,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if settings.DEBUG_MODE:
        print("--- DEBUG MODE ACTIVE ---")

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

    if settings.DEBUG_MODE:
        print("Char classes:", char_to_int.get_vocabulary())
        print(f"Longest phrase: {longest_label}. Len: {max_length}")


    def preprocess_sample(img_path, label):
        """Opens image and convert image to tensor of floats. Process characters from label to int format for CTC"""
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        img = 1.0 - img
        img = tf.image.resize_with_pad(img, 32, 256)
        img = 1.0 - img
        
        img = (img - 0.5) / 0.5 #Normalize

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

    if settings.DEBUG_MODE:
        print("Splits:  ",len(train_samples), len(val_samples), len(test_samples), len(samples))
        print("Batched: ",len(train_ds), len(val_ds), len(test_ds))

        for (sample, label) in train_ds.take(1):
            print("DS sample shape: ", sample.numpy().shape)
            print("Max, min values in sample: ", np.max(sample[0].numpy()), np.min(sample[0].numpy()))
            print("y_true: ", label.numpy())
            #plt.imshow(sample[0])
            #plt.title(tf.strings.reduce_join(int_to_char(label[0])).numpy().decode("UTF-8"))
            #plt.show()

    # MODEL
    model = build_model(input_shape, len(unique_chars) + 1)

    if settings.DEBUG_MODE:
        model.summary()

    # COMPILE
    model.compile(
        optimizer=keras.optimizers.Adam(), 
        loss= keras.losses.CTC(),
        metrics=[CharacterErrorRate(int_to_char), WordErrorRate(int_to_char)],
        run_eagerly=settings.EAGER_EXECUTION
    )

    # TRAINING
    history = model.fit(
        x=train_ds, 
        epochs=settings.EPOCHS, 
        validation_data=val_ds,
        callbacks=[
            ValidationLogCallback(val_ds, int_to_char),
            keras.callbacks.ModelCheckpoint(
                filepath=settings.CHECKPOINT_PATH,
                monitor="val_CER",
                verbose=1,
                save_best_only=True,
                mode="min"
            ),
        ],
    )

    with open(settings.HISTORY_PATH, "w") as file:
        json.dump(history.history, file, indent=4)

if __name__ == "__main__":
    main()