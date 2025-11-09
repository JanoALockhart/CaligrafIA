import os
from PIL import Image
import numpy as np
import pandas as pd
from data_augmentation import DatasetAugmentator

class RIMESDatasetAugmentator(DatasetAugmentator):
    def __init__(self, dataset_path, subfolder_name, train_split, val_split, dataloader):
        super().__init__(dataset_path, subfolder_name, train_split, val_split)
        self.dataloader = dataloader

    def split_dataset(self):
        word_paths, word_labels = self.dataloader.load_samples_tensor()

        total = len(word_labels)
        
        train_size = int(total * self.train_split)
        val_size = int(total * self.val_split)
        
        train_word_paths = word_paths[0:train_size]
        train_word_labels = word_labels[0:train_size]
        train_ds = (train_word_paths, train_word_labels)

        val_word_paths = word_paths[train_size:train_size+val_size]
        val_word_labels = word_labels[train_size:train_size+val_size]
        val_ds = (val_word_paths, val_word_labels)
        
        test_word_paths = word_paths[train_size+val_size:]
        test_word_labels = word_labels[train_size+val_size:]
        test_ds = (test_word_paths, test_word_labels)

        return train_ds, val_ds, test_ds
    
    def build_split_folder(self, ds_split, dest_folder, dest_labels_file):
        (words_paths, words_labels) = ds_split
        words_per_line = 5
        grouped_words = [words_paths[i:i+words_per_line] for i in range(0, len(words_paths), words_per_line)]
        grouped_labels = [words_labels[i:i+words_per_line] for i in range(0, len(words_labels), words_per_line)]
        
        new_img_paths = []
        new_labels = []
        for (words, labels) in zip(grouped_words, grouped_labels):
            file_name = self.get_file_name(words)
            relative_path = f"{dest_folder}/{file_name}.png"
            image_path = f"{self.base_path}{relative_path}"
            
            self._save_phrase_image(words, image_path)
            
            phrase_label = " ".join(labels)
            print(image_path, phrase_label)

            new_labels.append(phrase_label)
            new_img_paths.append(relative_path)

        df = pd.DataFrame({"path":new_img_paths, "label":new_labels})
        df.to_csv(f"{self.base_path}{dest_labels_file}", index = False)

    def _save_phrase_image(self, words, image_path):
        images = []
        space_between_words_px = 32
        for word_path in words:
            with Image.open(word_path) as image:
                img = image.convert("L")
                images.append(np.array(img, dtype=np.float32))
            
        max_height = max(img.shape[0] for img in images)
        total_width = sum(img.shape[1] + space_between_words_px for img in images)

        sentence_img = np.ones((max_height, total_width), dtype=np.float32) * 255.0

        x = 0
        for img in images:
            height, width = img.shape
            start_height = (max_height-height)//2
            sentence_img[start_height:start_height+height, x:x+width] = img
            x += width + space_between_words_px
        
        sentence_img = np.clip(sentence_img, 0, 255).astype(np.uint8)
        img_to_save = Image.fromarray(sentence_img)
        img_to_save.save(image_path, format="PNG")



    def get_file_name(self, words):
        first_name = words[0]
        last_split = first_name.split("/")[-1]
        file_name = last_split.split(".")[0]

        return file_name
