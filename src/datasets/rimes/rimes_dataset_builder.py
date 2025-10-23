import numpy as np
import tensorflow as tf
from PIL import Image
from datasets.dataset_builder import DatasetBuilder
from datasets.dataloader import Dataloader

class RIMESDatasetBuilder(DatasetBuilder):
    def __init__(self, dataloader:Dataloader, words_per_line = 5, space_between_words_px = 32):
        super().__init__(dataloader)
        self.words_per_line = words_per_line
        self.space_between_words_px = space_between_words_px

    def _load_concat_words(self, paths):
        images = []
        for p in paths.numpy():
            path_decoded = p.decode("utf-8")
            with Image.open(path_decoded) as image:
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
        
        sentence_img = sentence_img / 255.0
        sentence_img = np.expand_dims(sentence_img, axis=-1)
        
        return sentence_img

    def _combine_words(self, paths, labels):
        sentence_img = tf.py_function(self._load_concat_words, [paths], tf.float32)
        sentence_img.set_shape([None, None, 1])

        label = tf.strings.reduce_join(labels, separator=" ")

        return sentence_img, label

    def get_training_set(self):
        train_ds = super().get_training_set()
        train_ds = train_ds.batch(self.words_per_line, drop_remainder=True).shuffle(train_ds.cardinality()).map(self._combine_words)
        
        return train_ds

    def build_phrases(self, dataset):
        dataset = dataset.batch(self.words_per_line, drop_remainder=True).map(self._combine_words)
        return dataset
    
    def get_validation_set(self):
        val_ds = super().get_validation_set()
        val_ds = self.build_phrases(val_ds)
        return val_ds
    
    def get_test_set(self):
        test_ds = super().get_test_set()
        test_ds = self.build_phrases(test_ds)
        return test_ds
    
    def get_vocabulary(self):
        vocab = super().get_vocabulary()
        vocab.add(" ")
        return vocab
    
    def get_name(self):
        return "RIMES"
    
    