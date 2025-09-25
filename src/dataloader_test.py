import settings
import tensorflow as tf
import numpy as np
from dataloader import IAMLineDataloader
from main import char_to_int, int_to_char

def main():
    iam_dataloader = IAMLineDataloader(settings.IAM_PATH)
    (samples, labels) = iam_dataloader.load_samples_tensor()
    # print(samples)
    # print(labels)

    dataset = tf.data.Dataset.from_tensor_slices((samples, labels))

    for sample, label in dataset.take(1):
        print(sample.numpy(), label.numpy())

    print(len(dataset))

    total = len(dataset)
    train_split = int(0.95 * total)
    val_split = int(0.04 * total)
    test_split = total - train_split - val_split

    print(train_split, val_split, test_split, train_split + val_split + test_split)

    train_ds = dataset.take(train_split)
    val_ds = dataset.skip(train_split).take(val_split)
    test_ds = dataset.skip(train_split).skip(val_split)

    print(len(train_ds), len(val_ds), len(test_ds))

    label = "Hello"
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    encoded = char_to_int(label)
    decoded = int_to_char(encoded)
    print(f"Encode {label}: {encoded}")
    print(f"Decoded: {decoded}")

    print(f"The vocabulary is: {char_to_int.get_vocabulary()}")
    print(f"Size: {char_to_int.vocabulary_size()}") 

if __name__ == "__main__":
    main()