from abc import ABC, abstractmethod

import tensorflow as tf
from keras import layers
from data_augmentation import apply_augmentations
from datasets.dataset_builder import DatasetBuilder
import settings

class DatasetBroker(ABC):

    @abstractmethod
    def register_dataset_builder(self, dataset):
        pass

    @abstractmethod
    def sample_datasets(self):
        pass

    @abstractmethod
    def get_training_set(self):
        pass

    @abstractmethod
    def get_validation_set(self):
        pass

    @abstractmethod
    def get_test_set(self):
        pass

    @abstractmethod
    def get_encoding_function(self):
        pass

    @abstractmethod
    def get_decoding_function(self):
        pass

class DatasetBrokerImpl(DatasetBroker):
    def __init__(self, train_split_per, val_split_per, img_height, img_width, batch_size, data_augmentation = True):
        self.train_split = train_split_per
        self.val_split = val_split_per
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.dataset_builders:list[DatasetBuilder] = []

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.encoding_function = None
        self.decoding_function = None

    def register_dataset_builder(self, dataset_builder:DatasetBuilder):
        self.dataset_builders.append(dataset_builder)

    def sample_datasets(self):
        train_datasets = []
        val_datasets = []
        test_datasets = []
        vocabulary = set()

        for ds_builder in self.dataset_builders:
            ds_builder.set_splits(self.train_split, self.val_split)
            vocabulary = vocabulary.union(ds_builder.get_vocabulary())

            train_datasets.append(ds_builder.get_training_set())
            val_datasets.append(ds_builder.get_validation_set())
            test_datasets.append(ds_builder.get_test_set())

        vocabulary = sorted(vocabulary)
        
        self.encoding_function = layers.StringLookup(vocabulary=vocabulary, oov_token="[UNK]")
        self.decoding_function = layers.StringLookup(vocabulary=vocabulary, oov_token="[UNK]", invert=True)

        self.train_ds = tf.data.Dataset.sample_from_datasets(train_datasets)
        self.val_ds = tf.data.Dataset.sample_from_datasets(val_datasets)
        self.test_ds = tf.data.Dataset.sample_from_datasets(test_datasets)

    def _preprocess_sample(self, img, label):
        img = 1.0 - img
        img = tf.image.resize_with_pad(img, self.img_height, self.img_width)
        img = 1.0 - img
        
        img = (img - 0.5) / 0.5 #Normalize

        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        label = self.encoding_function(label)

        return img, label
    
    def _tf_augment(self, image, label):
        img_shape = image.shape
        image = apply_augmentations(image)
        image.set_shape(img_shape)

        return image, label

    def get_training_set(self):
        train_dataset = self.train_ds.map(self._preprocess_sample)
        if self.data_augmentation:
            train_dataset = train_dataset.map(self._tf_augment)
        train_dataset = train_dataset.padded_batch(self.batch_size, drop_remainder=True)
    
        return train_dataset

    def get_validation_set(self):
        return self.val_ds.map(self._preprocess_sample).padded_batch(self.batch_size)
    
    def get_test_set(self):
        return self.test_ds.map(self._preprocess_sample).padded_batch(self.batch_size)
    
    def get_encoding_function(self):
        return self.encoding_function
    
    def get_decoding_function(self):
        return self.decoding_function



    
