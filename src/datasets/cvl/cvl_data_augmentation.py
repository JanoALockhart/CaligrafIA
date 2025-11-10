import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from datasets.cvl.cvl_dataloader import CVLLineDataloader
from data_augmentation import DatasetAugmentator

# TODO: test
class CVLDatasetAugmentator(DatasetAugmentator):
    def __init__(self, dataset_path, subfolder_name, train_split, val_split, dataloader, img_shape = (512, 32)):
        super().__init__(dataset_path, subfolder_name, train_split, val_split)
        self.dataloader = dataloader

        self.img_shape = img_shape

    def split_dataset(self):
        img_paths, labels = self.dataloader.load_samples_tensor()
        total = len(labels)
        train_size = int(total * self.train_split)
        val_size = int(total * self.val_split)
        
        train_paths = img_paths[0:train_size]
        train_labels = labels[0:train_size]
        train_ds = (train_paths, train_labels)

        val_paths = img_paths[train_size:train_size+val_size]
        val_labels = labels[train_size:train_size+val_size]
        val_ds = (val_paths, val_labels)

        test_paths = img_paths[train_size+val_size:]
        test_labels = labels[train_size+val_size:]
        test_ds = (test_paths, test_labels)

        return train_ds, val_ds, test_ds

    def build_split_folder(self, ds_split, dest_folder, dest_labels_file):
        paths, labels = ds_split
        relative_paths = []
        for (img_path, label) in zip(paths, labels):
            file_name = self.get_file_name(img_path)
            file_name_png = f"{dest_folder}/{file_name}.png"
            relative_paths.append(file_name_png)

            print(file_name_png, label)

            with Image.open(img_path) as img:
                img = img.convert("L")
                x_freedom = np.random.random()
                img = ImageOps.pad(img, self.img_shape, color=(255, 255, 255), centering=(x_freedom, 0.5))
                img.save(f"{self.base_path}{file_name_png}", format="PNG")

        df = pd.DataFrame({"path":relative_paths, "label":labels})
        df.to_csv(f"{self.base_path}{dest_labels_file}", index = False)


    def get_file_name(self, img_path):
        path_split = img_path.split('/')
        file_name = path_split[-1]
        file_name_no_extention = file_name.split('.')[0]

        return file_name_no_extention
