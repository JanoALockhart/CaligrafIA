from abc import ABC, abstractmethod
from io import StringIO

import tensorflow as tf
from keras import layers
from tf_data_augmentation import apply_augmentations
from datasets.dataset_builder import DatasetBuilder
import settings

class DatasetBroker(ABC):

    @abstractmethod
    def register_training_dataset_builder(self, dataset):
        pass

    @abstractmethod
    def register_val_test_dataset_builders(self, dataset):
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
    def __init__(self, img_height, img_width, batch_size, data_augmentation = True):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.train_dataset_builders:list[DatasetBuilder] = []
        self.val_test_dataset_builders:list[DatasetBuilder] = []

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.encoding_function = None
        self.decoding_function = None

    def register_training_dataset_builder(self, dataset_builder:DatasetBuilder):
        self.train_dataset_builders.append(dataset_builder)

    def register_val_test_dataset_builders(self, dataset):
        self.val_test_dataset_builders.append(dataset)

    def sample_datasets(self):
        train_datasets = []
        val_datasets = []
        test_datasets = []
        vocabulary = set()

        for ds_builder in self.train_dataset_builders:
            vocabulary = vocabulary.union(ds_builder.get_vocabulary())
            train_datasets.append(ds_builder.get_training_set())

        for ds_builder in self.val_test_dataset_builders:
            vocabulary = vocabulary.union(ds_builder.get_vocabulary())
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
    
    def get_datasets_info(self):
        buffer = StringIO()
        
        buffer.write(f"=== Datasets === \n")
        
        total_train = 0

        for ds_builder in self.train_dataset_builders:
            buffer.write(f"--- {ds_builder.get_name()} TRAIN --- \n")
            train_split = ds_builder.get_train_split()*100
            
            train_size = ds_builder.get_training_set().cardinality()

            buffer.write(f"Train: {train_split: .2f}% / {train_size} images\n")
            
            total_train += train_size
            
        total_val = 0
        total_test = 0
        for ds_builder in self.val_test_dataset_builders:
            buffer.write(f"--- {ds_builder.get_name()} VAL and TEST --- \n")
            val_split = ds_builder.get_val_split()*100
            test_split = ds_builder.get_test_split()*100

            val_size = ds_builder.get_validation_set().cardinality()
            test_size = ds_builder.get_test_set().cardinality()

            buffer.write(f"Validation: {val_split: .2f}% / {val_size} images \n")
            buffer.write(f"Test: {test_split: .2f}% / {test_size} images \n")

            total_val += val_size
            total_test += test_size

        total_samples = total_train + total_val + total_test
        total_train_split = total_train / total_samples * 100
        total_val_split = total_val / total_samples * 100
        total_test_split = total_test / total_samples * 100

        buffer.write(f"===TOTAL===\n")
        buffer.write(f"Train: {total_train_split: .2f}% / {total_train} images\n")
        buffer.write(f"Validation: {total_val_split: .2f}% / {total_val} images\n")
        buffer.write(f"Test: {total_test_split: .2f}% / {total_test} images\n")

        vocab = "".join(self.encoding_function.get_vocabulary())
        buffer.write(f"===VOCABULARY===\n")
        buffer.write(f"{vocab}")

        return buffer.getvalue()

    
