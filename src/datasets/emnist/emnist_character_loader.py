import tensorflow as tf
import tensorflow_datasets as tfds

class EMNISTCharacterDataset():
    def __init__(self):
        self.ds_train, self.info = tfds.load(
            "emnist",
            split="train[:90%]",
            as_supervised=True,
            with_info=True
        )

        self.ds_val = tfds.load(
            "emnist",
            split="train[90%:]",
            as_supervised=True
        )

        self.ds_test = tfds.load(
            "emnist",
            split="test",
            as_supervised=True,
        )
        self.characters = tf.constant(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))


        print("Loaded Sets: ", )
        

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
    
    def get_info(self):
        return self.info