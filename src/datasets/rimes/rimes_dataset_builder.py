import numpy as np
import tensorflow as tf
from PIL import Image
from datasets.dataset_builder import DatasetBuilder
from datasets.dataloader import Dataloader

class RIMESDatasetBuilder(DatasetBuilder):
    def __init__(self, dataloader:Dataloader, words_per_line = 5, space_between_words_px = 32):
        self.dataloader = dataloader
        self.words_per_line = words_per_line
        self.space_between_words_px = space_between_words_px
        self.train_split_per = None
        self.val_split_per = None
        self.samples, self.labels = dataloader.load_samples_tensor()
        self.total = len(self.samples)

    def set_splits(self, train_split_per, val_split_per):
        self.train_split_per = train_split_per
        self.val_split_per = val_split_per

    def _load_concat_words(self, paths):
        images = []
        for p in paths:
            with Image.open(p.numpy().decode("utf-8")) as image:
                img = image.convert("L")
                images.append(np.array(img, dtype=np.float32))

        
        max_height = max(img.shape[0] for img in images)
        total_width = sum(img.shape[1] + self.space_between_words_px for img in images)

        sentence_img = np.ones((max_height, total_width), dtype=np.float32) * 255.0

        x = 0
        for img in images:
            height, width = img.shape
            start_height = (max_height-height)//2
            sentence_img[start_height:start_height+height, x:x+width] = img
            x += width + self.space_between_words_px
        
        sentence_img = np.expand_dims(sentence_img, axis=-1)

        return sentence_img

    def _combine_words(self, paths, labels):
        sentence_img = tf.py_function(self._load_concat_words, [paths], tf.float32)
        sentence_img.set_shape([None, None, 1])

        label = tf.strings.reduce_join(labels, separator=" ")

        return sentence_img, label

    def get_training_set(self):
        train_split = int(self.train_split_per * self.total)
        train_samples = self.samples[0:train_split]
        train_labels = self.labels[0:train_split]
        train_ds = tf.data.Dataset.from_tensor_slices((train_samples, train_labels))
        train_ds = train_ds.batch(self.words_per_line, drop_remainder=True).map(self._combine_words)
        
        return train_ds
    
    def get_validation_set(self):
        train_split = int(self.train_split_per * self.total)
        val_split = int(self.val_split_per * self.total)
        val_samples = self.samples[train_split:train_split+val_split]
        val_labels = self.labels[train_split:train_split+val_split]
        val_ds = tf.data.Dataset.from_tensor_slices((val_samples, val_labels))
        val_ds = val_ds.batch(self.words_per_line, drop_remainder=True).map(self._combine_words)
        
        return val_ds
    
    def get_test_set(self):
        train_split = int(self.train_split_per * self.total)
        val_split = int(self.val_split_per * self.total)
        test_samples = self.samples[train_split+val_split:]
        test_labels = self.labels[train_split+val_split:]
        test_ds = tf.data.Dataset.from_tensor_slices((test_samples, test_labels))
        test_ds = test_ds.batch(self.words_per_line, drop_remainder=True).map(self._combine_words)

        return test_ds
    
    def get_vocabulary(self):
        return self.dataloader.get_vocabulary()