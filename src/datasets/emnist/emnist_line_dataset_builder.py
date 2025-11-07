import tensorflow as tf
from datasets.dataset_builder import DatasetBuilder
from datasets.emnist.emnist_character_loader import EMNISTCharacterDataset

class EMNISTLineDatasetBuilder(DatasetBuilder):
    def __init__(self, dataloader:EMNISTCharacterDataset, chars_per_line = 18):
        self.dataloader = dataloader
        self.chars_per_line = chars_per_line
        self.place_width = 28
        self.space_probability = 0.25

    def _synthesize_random_line(self, batch_img_char, batch_label_char):
        row_image, row_label = tf.py_function(self._synthesize_random_line_py, [batch_img_char, batch_label_char], (tf.float32, tf.string))
        row_image.set_shape([None, None, 1])
        row_label.set_shape([])

        return row_image, row_label

    def _synthesize_random_line_py(self, batch_img_char, batch_label_char):
        batch_img_char = tf.image.convert_image_dtype(batch_img_char, tf.float32)
        batch_img_char = 1 - batch_img_char
        height = tf.shape(batch_img_char)[1]

        spaces_template = self._generate_spaces_template()

        new_line = []
        new_label = []

        for i in range(self.chars_per_line):
            is_space = spaces_template[i] > 0
            if is_space:
                width = spaces_template[i]
                space = tf.ones([height, width, 1], tf.float32)
                new_line.append(space)
                new_label.append(tf.constant(' '))
            else:
                img = batch_img_char[i]
                new_line.append(img)
                new_label.append(batch_label_char[i])               

        row_image = tf.concat(new_line, axis=1)
        row_label = tf.strings.reduce_join(new_label)
        return row_image, row_label

    def _generate_spaces_template(self):
        mask = tf.random.uniform([self.chars_per_line], 0, 1)
        mask = tf.cast(mask < self.space_probability, tf.float32)

        random_values = tf.random.uniform([self.chars_per_line], 0.0, 1.0)
        space_lengths = mask * random_values

        total = tf.reduce_sum(space_lengths)
        space_length_percentaje = tf.math.divide_no_nan(space_lengths, total)

        amount_spaces = tf.reduce_sum(mask)
        space_widths = space_length_percentaje * (self.place_width * amount_spaces)
        space_widths = tf.cast(space_widths, tf.int32)

        return space_widths


    def get_training_set(self):
        train_ds = self.dataloader.get_training_set()
        train_ds = train_ds.batch(self.chars_per_line, drop_remainder=True).map(self._synthesize_random_line)

        return train_ds

    def get_validation_set(self):
        val_ds = self.dataloader.get_validation_set()
        val_ds = val_ds.batch(self.chars_per_line, drop_remainder=True).map(self._synthesize_random_line)

        return val_ds

    def get_test_set(self):
        test_ds = self.dataloader.get_test_set()
        test_ds = test_ds.batch(self.chars_per_line, drop_remainder=True).map(self._synthesize_random_line)

        return test_ds

    def get_vocabulary(self):
        return set(" 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    def get_name(self):
        return "EMNIST Lines"
    
    def get_train_split(self):
        return self.dataloader.get_train_split()
    
    def get_val_split(self):
        return self.dataloader.get_val_split()
    
    def get_test_split(self):
        return 1 - self.get_val_split() - self.get_train_split()