import os
from PIL import Image
import numpy as np
import pandas as pd
from data_augmentation import build_data_aug_folder
from datasets.rimes.rimes_dataloader import RIMESWordsDataloader


def augment_RIMES(rimes_path, name="lines_png", train_split = 0.6, val_split = 0.2):
    dataloader = RIMESWordsDataloader(rimes_path)
    word_paths, word_labels = dataloader.load_samples_tensor()

    total = len(word_labels)
    
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    
    train_word_paths = word_paths[0:train_size]
    train_word_labels = word_labels[0:train_size]

    val_word_paths = word_paths[train_size:train_size+val_size]
    val_word_labels = word_labels[train_size:train_size+val_size]
    
    test_word_paths = word_paths[train_size+val_size:]
    test_word_labels = word_labels[train_size+val_size:]

    base_path = f"{rimes_path}/{name}"
    TRAIN_FOLDER = f"/train"
    TRAIN_DATA_AUG_FOLDER = f"/train_data_aug"
    VAL_FOLDER = f"/val"
    TEST_FOLDER = f"/test"
    LABELS_FOLDER = f"/labels"

    subfolders = [TRAIN_FOLDER, TRAIN_DATA_AUG_FOLDER, VAL_FOLDER, TEST_FOLDER, LABELS_FOLDER]

    create_folders(base_path, subfolders)

    train_labels_file = f"{LABELS_FOLDER}/train.csv"
    build_split_folder(train_word_paths, train_word_labels, base_path, TRAIN_FOLDER, train_labels_file)
    build_split_folder(val_word_paths, val_word_labels, base_path, VAL_FOLDER, f"{LABELS_FOLDER}/val.csv")
    build_split_folder(test_word_paths, test_word_labels, base_path, TEST_FOLDER, f"{LABELS_FOLDER}/test.csv")

    data_aug_labels_file = f"{LABELS_FOLDER}/train_da.csv"
    build_split_folder(train_word_paths, train_word_labels, base_path, TRAIN_DATA_AUG_FOLDER, data_aug_labels_file)
    build_data_aug_folder(base_path, train_labels_file, TRAIN_DATA_AUG_FOLDER, data_aug_labels_file)

def build_split_folder(words_paths, words_labels, base_path, image_folder, labels_file):
    words_per_line = 5
    grouped_words = [words_paths[i:i+words_per_line] for i in range(0, len(words_paths), words_per_line)]
    grouped_labels = [words_labels[i:i+words_per_line] for i in range(0, len(words_labels), words_per_line)]
    
    new_img_paths = []
    new_labels = []
    for (words, labels) in zip(grouped_words, grouped_labels):
        file_name = get_file_name(words)
        relative_path = f"{image_folder}/{file_name}.png"
        image_path = f"{base_path}{relative_path}"
        
        
        save_phrase_image(words, image_path)
        
        phrase_label = " ".join(labels)
        print(image_path, phrase_label)

        new_labels.append(phrase_label)
        new_img_paths.append(relative_path)

    df = pd.DataFrame({"path":new_img_paths, "label":new_labels})
    df.to_csv(f"{base_path}{labels_file}", index = False)

def save_phrase_image(words, image_path):
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



def get_file_name(words):
    first_name = words[0]
    last_split = first_name.split("/")[-1]
    file_name = last_split.split(".")[0]

    return file_name

def create_folders(base_path, subfolders):
    for sub in subfolders:
        os.makedirs(base_path + sub, exist_ok=True)
