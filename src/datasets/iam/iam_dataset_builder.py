import tensorflow as tf
from datasets.dataloader import Dataloader
from datasets.dataset_builder import DatasetBuilder

class IAMDatasetBuilder(DatasetBuilder):
    def __init__(self, dataloader:Dataloader):
        super().__init__(dataloader)
    
    def _load_image(self, img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label

    def get_training_set(self):
        train_ds = super().get_training_set()
        train_ds = train_ds.shuffle(train_ds.cardinality()).map(self._load_image)
        return train_ds
    
    def get_validation_set(self):
        val_ds = super().get_validation_set()
        val_ds = val_ds.map(self._load_image)
        return val_ds
    
    def get_test_set(self):
        test_ds = super().get_test_set()
        test_ds = test_ds.map(self._load_image)
        return test_ds

        