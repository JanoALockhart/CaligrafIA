import tensorflow as tf
import tensorflow_datasets as tfds

class EMNISTCharacterDataset():
    def __init__(self, train_split=0.6, val_split=0.2):
        self.dataset, self.info = tfds.load(
            "emnist",
            split="train+test",
            as_supervised=True,
            with_info=True
        )

        self.train_split_per = train_split
        self.val_split_per = val_split

        self.train_size = self.dataset.cardinality().numpy() * train_split
        self.val_size = self.dataset.cardinality().numpy() * val_split
        
        self.ds_train = self.dataset.take(self.train_size)
        self.ds_val = self.dataset.skip(self.train_size).take(self.val_size)
        self.ds_test = self.dataset.skip(self.train_size+self.val_size)

        self.characters = tf.constant(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))

    def _fix_orientation(self, image, label):
        image = tf.image.transpose(image)

        return image, label
    
    def _decode_label(self, image, label):
        byte_char =  tf.gather(self.characters, label)
        return image, byte_char

    def get_training_set(self):
        return self.ds_train.map(self._fix_orientation).map(self._decode_label)
    
    def get_validation_set(self):
        return self.ds_val.map(self._fix_orientation).map(self._decode_label)
    
    def get_test_set(self):
        return self.ds_test.map(self._fix_orientation).map(self._decode_label)
    
    def get_name(self):
        return "EMNIST"

    def get_train_split(self):
        return self.train_split_per

    def get_val_split(self):
        return self.val_split_per
    
    def get_info(self):
        return self.info