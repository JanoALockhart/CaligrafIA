from abc import ABC
from abc import abstractmethod

import tensorflow as tf

from datasets.dataloader import Dataloader

class DatasetBuilder(ABC):

    def __init__(self, dataloader:Dataloader, train_split, val_split):
        self.dataloader = dataloader
        self.train_split_per = train_split
        self.val_split_per = val_split
        self.samples, self.labels = dataloader.load_samples_tensor()
        self.total = len(self.samples)

    @abstractmethod
    def get_training_set(self):
        train_split = int(self.train_split_per * self.total)
        train_samples = self.samples[0:train_split]
        train_labels = self.labels[0:train_split]
        train_ds = tf.data.Dataset.from_tensor_slices((train_samples, train_labels))

        return train_ds

    @abstractmethod
    def get_validation_set(self):
        train_split = int(self.train_split_per * self.total)
        val_split = int(self.val_split_per * self.total)
        val_samples = self.samples[train_split:train_split+val_split]
        val_labels = self.labels[train_split:train_split+val_split]
        val_ds = tf.data.Dataset.from_tensor_slices((val_samples, val_labels))

        return val_ds

    @abstractmethod
    def get_test_set(self):
        train_split = int(self.train_split_per * self.total)
        val_split = int(self.val_split_per * self.total)
        test_samples = self.samples[train_split+val_split:]
        test_labels = self.labels[train_split+val_split:]
        test_ds = tf.data.Dataset.from_tensor_slices((test_samples, test_labels))

        return test_ds

    def get_vocabulary(self):
        return self.dataloader.get_vocabulary()
    
    @abstractmethod
    def get_name(self):
        return "NO_NAME"