import tensorflow as tf
from datasets.dataloader import Dataloader
from datasets.dataset_builder import DatasetBuilder

class IAMDatasetBuilder(DatasetBuilder):
    def __init__(self, dataloader:Dataloader):
        self.dataloader = dataloader
        self.train_split_per = None
        self.val_split_per = None
        self.samples, self.labels = dataloader.load_samples_tensor()
        self.total = len(self.samples)
    
    def _load_image(self, img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label
    
    def set_splits(self, train_split_per, val_split_per):
        self.train_split_per = train_split_per
        self.val_split_per = val_split_per

    def get_training_set(self):
        train_split = int(self.train_split_per * self.total)
        train_samples = self.samples[0:train_split]
        train_labels = self.labels[0:train_split]
        train_ds = tf.data.Dataset.from_tensor_slices((train_samples, train_labels))
        train_ds = train_ds.shuffle(train_ds.cardinality()).map(self._load_image)
        return train_ds
    
    def get_validation_set(self):
        train_split = int(self.train_split_per * self.total)
        val_split = int(self.val_split_per * self.total)
        val_samples = self.samples[train_split:train_split+val_split]
        val_labels = self.labels[train_split:train_split+val_split]
        val_ds = tf.data.Dataset.from_tensor_slices((val_samples, val_labels))
        val_ds = val_ds.map(self._load_image)
        return val_ds
    
    def get_test_set(self):
        train_split = int(self.train_split_per * self.total)
        val_split = int(self.val_split_per * self.total)
        test_samples = self.samples[train_split+val_split:]
        test_labels = self.labels[train_split+val_split:]
        test_ds = tf.data.Dataset.from_tensor_slices((test_samples, test_labels))
        test_ds = test_ds.map(self._load_image)
        return test_ds
    
    def get_vocabulary(self):
        return self.dataloader.get_vocabulary()

        