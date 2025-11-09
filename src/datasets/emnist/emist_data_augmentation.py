import os
import numpy as np
from PIL import Image
import pandas as pd
import data_augmentation as da
from datasets.emnist.emnist_character_loader import EMNISTCharacterDataset
from datasets.emnist.emnist_line_dataset_builder import EMNISTLineDatasetBuilder


def augment_EMNIST_lines(
    emnist_path,
    name = "emnist_lines",
    chars_per_line = 18,
    train_split = 0.6,
    val_split = 0.2
):
    
    base_path = f"{emnist_path}/{name}"
    TRAIN_FOLDER = f"/train"
    TRAIN_DATA_AUG_FOLDER = f"/train_data_aug"
    VAL_FOLDER = f"/val"
    TEST_FOLDER = f"/test"
    LABELS_FOLDER = f"/labels"

    subfolders = [TRAIN_FOLDER, TRAIN_DATA_AUG_FOLDER, VAL_FOLDER, TEST_FOLDER, LABELS_FOLDER]
    create_folders(base_path, subfolders)

    loader = EMNISTCharacterDataset(train_split, val_split)
    dataset = EMNISTLineDatasetBuilder(loader)

    train_ds = loader.get_training_set().batch(chars_per_line, drop_remainder=True).map(dataset._synthesize_random_line)
    val_ds = loader.get_validation_set().batch(chars_per_line, drop_remainder=True).map(dataset._synthesize_random_line)
    test_ds = loader.get_test_set().batch(chars_per_line, drop_remainder=True).map(dataset._synthesize_random_line)

    train_labels_file = f"{LABELS_FOLDER}/train.csv"
    build_slice_folder(train_ds, base_path, TRAIN_FOLDER, train_labels_file)
    build_slice_folder(val_ds, base_path, VAL_FOLDER, f"{LABELS_FOLDER}/val.csv")
    build_slice_folder(test_ds, base_path, TEST_FOLDER, f"{LABELS_FOLDER}/test.csv")

    #Build augmented
    data_aug_labels_file = f"{LABELS_FOLDER}/train_da.csv"
    build_slice_folder(train_ds, base_path, TRAIN_DATA_AUG_FOLDER, data_aug_labels_file)
    da.build_data_aug_folder(base_path, train_labels_file, TRAIN_DATA_AUG_FOLDER, data_aug_labels_file)

def build_slice_folder(dataset, base_path, dest_folder, labels_file_path):
    dataset = dataset.take(5) #REMOVE
    
    new_img_paths = []
    new_labels = []
    for i, (line, label) in enumerate(dataset):
        file_name = f"{i:05d}.png"
        relative_path = f"{dest_folder}/{file_name}"
        label = label.numpy().decode("utf-8")

        img = np.squeeze(line.numpy())
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(f"{base_path}{relative_path}")

        new_img_paths.append(relative_path)
        new_labels.append(label)

        print(relative_path, label)
    
    df = pd.DataFrame({"path":new_img_paths, "label":new_labels})
    df.to_csv(f"{base_path}{labels_file_path}", index = False)
        

def create_folders(base_path, subfolders):
    for sub in subfolders:
        os.makedirs(base_path + sub, exist_ok=True)    