import os
import numpy as np
from PIL import Image
import pandas as pd
from datasets.emnist.emnist_character_loader import EMNISTCharacterDataset
from datasets.emnist.emnist_line_dataset_builder import EMNISTLineDatasetBuilder
from data_augmentation import DatasetAugmentator

class EMNISTDatasetAugmentator(DatasetAugmentator):
    def __init__(self, dataset_path, subfolder_name, train_split, val_split, chars_per_line = 18):
        super().__init__(dataset_path, subfolder_name, train_split, val_split)
        self.chars_per_line = chars_per_line
        self.loader = EMNISTCharacterDataset(train_split, val_split)
        self.dataset = EMNISTLineDatasetBuilder(self.loader)
    
    def split_dataset(self):
        train_ds = self.loader.get_training_set().batch(self.chars_per_line, drop_remainder=True).map(self.dataset._synthesize_random_line)
        val_ds = self.loader.get_validation_set().batch(self.chars_per_line, drop_remainder=True).map(self.dataset._synthesize_random_line)
        test_ds = self.loader.get_test_set().batch(self.chars_per_line, drop_remainder=True).map(self.dataset._synthesize_random_line)

        return train_ds, val_ds, test_ds

    def build_split_folder(self, ds_split, dest_folder, dest_labels_file):
        new_img_paths = []
        new_labels = []
        for i, (line, label) in enumerate(ds_split):
            file_name = f"{i:05d}.png"
            relative_path = f"{dest_folder}/{file_name}"
            label = label.numpy().decode("utf-8")

            img = np.squeeze(line.numpy())
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(f"{self.base_path}{relative_path}")

            new_img_paths.append(relative_path)
            new_labels.append(label)

            print(relative_path, label)
        
        df = pd.DataFrame({"path":new_img_paths, "label":new_labels})
        df.to_csv(f"{self.base_path}{dest_labels_file}", index = False)
