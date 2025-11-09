from abc import ABC, abstractmethod
import os

import numpy as np
import pandas as pd
from PIL import Image
import data_augmentation as da
import datasets.cvl.cvl_data_augmentation as cvl_da
import datasets.rimes.rimes_data_augmentation as rimes_da
import datasets.emnist.emist_data_augmentation as emnist_da
import settings

def augment_cvl():
    cvl_da.augment_CVL(
        cvl_path = settings.CVL_PATH,
        name = "lines_png",
        train_split = settings.TRAIN_SPLIT, 
        val_split=settings.VAL_SPLIT, 
    )

def augment_rimes():
    rimes_da.augment_RIMES(
        rimes_path=settings.RIMES_PATH,
        name="lines_png",
        train_split=settings.TRAIN_SPLIT,
        val_split=settings.VAL_SPLIT
    )

def augment_EMNIST():
    emnist_da.augment_EMNIST_lines(
        emnist_path = settings.EMNIST_PATH, #TODO Define
        name="emnist_lines",
        chars_per_line=18,
        train_split=settings.TRAIN_SPLIT,
        val_split=settings.VAL_SPLIT
    )

class DatasetAugmentator(ABC):
    def __init__(self, dataset_path, subfolder_name = "augmented"):
        self.base_path = f"{dataset_path}/{subfolder_name}"
        
        self.TRAIN_FOLDER = "/train"
        self.VALIDATION_FOLDER = "/val"
        self.TEST_FOLDER = "/test"
        self.DATA_AUG_FOLDER = "/train_aug"

        self.LABEL_FOLDER = "/labels"
        self.TRAIN_LABELS_FILE = f"{self.LABEL_FOLDER}/train.csv"
        self.VAL_LABELS_FILE = f"{self.LABEL_FOLDER}/val.csv"
        self.TEST_LABELS_FILE = f"{self.LABEL_FOLDER}/test.csv"
        self.DATA_AUG_LABELS_FILE = f"{self.LABEL_FOLDER}/train_da.csv"


    def augment_dataset(self):
        # Template method pattern
        self.create_folder_structure()
        train_ds, val_ds, test_ds = self.split_dataset()
        self.build_split_folder(train_ds, self.TRAIN_FOLDER, self.TRAIN_LABELS_FILE)
        self.build_split_folder(val_ds, self.VALIDATION_FOLDER, self.VAL_LABELS_FILE)
        self.build_split_folder(test_ds, self.TEST_FOLDER, self.TEST_LABELS_FILE)

        self.build_split_folder(train_ds, self.DATA_AUG_FOLDER, self.DATA_AUG_LABELS_FILE)
        self.build_data_aug_folder(self.TRAIN_LABELS_FILE, self.DATA_AUG_FOLDER, self.DATA_AUG_LABELS_FILE)

    @abstractmethod
    def split_dataset(self):
        # Each subclass must define the splits percentajes
        pass

    @abstractmethod
    def build_split_folder(ds_split, dest_folder, dest_labels_file):
        pass

    def build_data_aug_folder(self, source_labels_file, dest_folder, dest_labels_file):
        df = pd.read_csv(self.base_path + source_labels_file)

        new_image_paths = []
        new_labels = []

        for row in df.itertuples(index=False):

            img_path = self.base_path + row.path
            with Image.open(img_path) as img:
                img = img.convert("L")

                for i in range(5):
                    img_aug = np.array(img, dtype=np.float32) / 255.0
                    img_aug = da.apply_all_techniques(img_aug)
                    img_aug = np.clip(img_aug * 255, 0, 255).astype(np.uint8)

                    img_aug = Image.fromarray(img_aug)
                    file_name = row.path.split("/")[-1].split(".")[0]
                    relative_path_img_aug = f"{dest_folder}/{file_name}-{i}.png"
                    img_aug.save(f"{self.base_path}{relative_path_img_aug}", format="PNG")

                    new_image_paths.append(relative_path_img_aug)
                    new_labels.append(row.label)

                    print(relative_path_img_aug, row.label)

        dest_df = pd.DataFrame({"path":new_image_paths, "label":new_labels})
        dest_df.to_csv(f"{self.base_path}{dest_labels_file}", mode="a", header=False, index=False)

    def create_folder_structure(self):
        subfolders = [self.TRAIN_FOLDER, self.VALIDATION_FOLDER, self.TEST_FOLDER, self.DATA_AUG_FOLDER]
        for sub in subfolders:
            os.makedirs(self.base_path + sub, exist_ok=True)


if __name__ == "__main__":
    #augment_cvl()
    #augment_rimes()
    augment_EMNIST()


