from PIL import Image
import numpy as np
import tensorflow as tf
from datasets.dataloader import Dataloader
from datasets.dataset_builder import DatasetBuilder

class CVLDatasetBuilder(DatasetBuilder):
    def __init__(self, dataloader:Dataloader):
        super().__init__(dataloader)

    def _load_tif(self, path):
        with Image.open(path.numpy().decode("utf-8")) as image:
            img = image.convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0

        arr = np.expand_dims(arr, axis=-1)
        return arr
    
    def _load_image(self, path, label):
        img = tf.py_function(self._load_tif, [path], tf.float32)
        img.set_shape([None, None, 1])
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