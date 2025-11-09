import tensorflow as tf
import pandas as pd
from datasets.dataset_builder import DatasetBuilder


class CustomAugmentedDatasetBuilder(DatasetBuilder):
    def __init__(self, dataset_path, subfolder_name = "lines_png", data_augmentation = False):
        self.dataset_path = dataset_path
        self.base_path = f"{dataset_path}/{subfolder_name}"
        self.data_augmentation = data_augmentation

        self.LABEL_FOLDER = "/labels"
        self.TRAIN_LABELS_FILE = f"{self.LABEL_FOLDER}/train.csv"
        self.VAL_LABELS_FILE = f"{self.LABEL_FOLDER}/val.csv"
        self.TEST_LABELS_FILE = f"{self.LABEL_FOLDER}/test.csv"
        self.DATA_AUG_LABELS_FILE = f"{self.LABEL_FOLDER}/train_da.csv"

        self.vocabulary = self._build_vocabulary()
        self.train_split, self.val_split, self.test_split = self._split_percentajes()
    
    def _split_percentajes(self):
        train_df = pd.read_csv(f"{self.base_path}{self.TRAIN_LABELS_FILE}")
        val_df = pd.read_csv(f"{self.base_path}{self.VAL_LABELS_FILE}")
        test_df = pd.read_csv(f"{self.base_path}{self.TEST_LABELS_FILE}")

        total = len(train_df)+len(val_df)+len(test_df)
        return len(train_df)/total, len(val_df)/total, len(test_df)/total

    def _build_vocabulary(self):
        vocab = set()

        vocab = vocab.union(self._build_split_vocabulary(self.TRAIN_LABELS_FILE))
        vocab = vocab.union(self._build_split_vocabulary(self.VAL_LABELS_FILE))
        vocab = vocab.union(self._build_split_vocabulary(self.TEST_LABELS_FILE))

        return vocab
    
    def _build_split_vocabulary(self, labels_file):
        vocab = set()

        df = pd.read_csv(f"{self.base_path}{labels_file}")
        all_text = "".join(df["label"].astype(str))
        vocab = set(all_text)

        return vocab

    def get_train_split(self):
        return self.train_split

    def get_val_split(self):
        return self.val_split

    def get_test_split(self):
        return self.test_split

    def get_name(self):
        return self.dataset_path.split("/")[-1]
    
    def get_vocabulary(self):
        return self.vocabulary
    
    def _add_base_path(self, path, label):
        base = tf.constant(self.base_path)
        new_path = tf.strings.join([base, path])
        return new_path, label
    
    def _load_image(self, img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, label

    def get_training_set(self):
        if self.data_augmentation:
            df = pd.read_csv(f"{self.base_path}{self.DATA_AUG_LABELS_FILE}")
        else:
            df = pd.read_csv(f"{self.base_path}{self.TRAIN_LABELS_FILE}")
        paths = df["path"].tolist()
        labels = df["label"].tolist()
        labels = list(map(str, labels))

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.shuffle(dataset.cardinality()).map(self._add_base_path).map(self._load_image)
        return dataset
    
    def get_validation_set(self):
        df = pd.read_csv(f"{self.base_path}{self.VAL_LABELS_FILE}")
        paths = df["path"].tolist()
        labels = df["label"].tolist()
        return tf.data.Dataset.from_tensor_slices((paths, labels)).map(self._add_base_path).map(self._load_image)
    
    def get_test_set(self):
        df = pd.read_csv(f"{self.base_path}{self.TEST_LABELS_FILE}")
        paths = df["path"].tolist()
        labels = df["label"].tolist()
        return tf.data.Dataset.from_tensor_slices((paths, labels)).map(self._add_base_path).map(self._load_image)
    
