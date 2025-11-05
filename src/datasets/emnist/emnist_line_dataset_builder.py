import tensorflow as tf
from datasets.dataset_builder import DatasetBuilder
from datasets.emnist.emnist_character_loader import EMNISTCharacterDataset

class EMNISTLineDatasetBuilder(DatasetBuilder):
    def __init__(self, dataloader:EMNISTCharacterDataset, letters_per_line = 18):
        self.dataloader = dataloader
        self.letters_per_line = letters_per_line
        self.space_width = 28

    def _synthesize_line(self, batch_img_char, batch_label_char):
        batch_img_char = tf.image.convert_image_dtype(batch_img_char, tf.float32)
        batch_img_char = 1 - batch_img_char
        height = tf.shape(batch_img_char)[1]
        space = tf.ones([height, self.space_width, 1], tf.float32)
        letters = tf.unstack(batch_img_char, axis=0)
        line = []
        for i, img in enumerate(letters):
            line.append(img)
            if i < len(letters) - 1:
                line.append(space)
        row_image = tf.concat(line, axis=1)

        row_label = tf.strings.reduce_join(batch_label_char, separator=' ')

        return row_image, row_label


    def get_training_set(self):
        train_ds = self.dataloader.get_training_set()
        train_ds = train_ds.batch(self.letters_per_line, drop_remainder=True).map(self._synthesize_line)

        return train_ds

    def get_validation_set(self):
        val_ds = self.dataloader.get_validation_set()
        val_ds = val_ds.batch(self.letters_per_line, drop_remainder=True).map(self._synthesize_line)

        return val_ds

    def get_test_set(self):
        test_ds = self.dataloader.get_test_set()
        test_ds = test_ds.batch(self.letters_per_line, drop_remainder=True).map(self._synthesize_line)

        return test_ds

    def get_vocabulary(self):
        return set(" 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    def get_name(self):
        return "EMNIST Lines"