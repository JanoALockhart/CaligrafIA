import tensorflow as tf
from datasets.dataset_builder import DatasetBuilder
import tensorflow_datasets as tfds

class EMNISTDatasetBuilder(DatasetBuilder):
    def __init__(self):
        self.ds_train, self.info = tfds.load(
            "emnist/letters",
            split="train",
            as_supervised=True,
            with_info=True
        )

        self.ds_test = tfds.load(
            "emnist/letters",
            split="test",
            as_supervised=True,
        )

        

    def _fix_orientation(self, image, label): # TODO: map to datasets
        image = tf.image.transpose(image)
        image = tf.image.flip_left_right(image)


    def get_training_set(self):
        return self.ds_train
    
    def get_validation_set(self):
        pass #TODO
    
    def get_test_set(self):
        return self.ds_test
    
    def get_name(self):
        return "EMNIST"
    
    def get_info(self):
        return self.info