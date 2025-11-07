import os
from PIL import Image
import numpy as np
import pandas as pd
from datasets.cvl.cvl_dataloader import CVLLineDataloader
import data_augmentation as da

def augment_CVL(
        cvl_path,
        name = "lines_png",
        train_split = 0.6, 
        val_split = 0.2,
    ):

    dataloader = CVLLineDataloader(cvl_path)
    img_paths, labels = dataloader.load_samples_tensor()

    #REMOVE THIS IS TO CUT IMAGES FOR TESTING
    img_paths = img_paths[:10]
    labels = labels[:10]
    #

    total = len(labels)
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    
    train_paths = img_paths[0:train_size]
    train_labels = labels[0:train_size]

    val_paths = img_paths[train_size:train_size+val_size]
    val_labels = labels[train_size:train_size+val_size]
    
    test_paths = img_paths[train_size+val_size:]
    test_labels = labels[train_size+val_size:]

    #create folder structure
    base_path = f"{cvl_path}/{name}"
    TRAIN_FOLDER = f"/train"
    TRAIN_DATA_AUG_FOLDER = f"/train_data_aug"
    VAL_FOLDER = f"/val"
    TEST_FOLDER = f"/test"
    LABELS_FOLDER = f"/labels"

    #create_folders(base_path, TRAIN_FOLDER, TRAIN_DATA_AUG_FOLDER, VAL_FOLDER, TEST_FOLDER, LABELS_FOLDER)

    build_split_folder(train_paths, train_labels, base_path, TRAIN_FOLDER, f"{LABELS_FOLDER}/train.csv")
    build_split_folder(val_paths, val_labels, base_path, VAL_FOLDER, labels_file_path=f"{LABELS_FOLDER}/val.csv")
    build_split_folder(test_paths, test_labels, base_path, TEST_FOLDER, labels_file_path=f"{LABELS_FOLDER}/test.csv")
    build_split_folder(train_paths, train_labels, base_path, TRAIN_DATA_AUG_FOLDER, f"{LABELS_FOLDER}/train_da.csv")
    build_data_aug_folder(train_paths, train_labels, base_path, TRAIN_DATA_AUG_FOLDER, labels_file_path=f"{LABELS_FOLDER}/train_da.csv")

def create_folders(base_path, TRAIN_FOLDER, TRAIN_DATA_AUG_FOLDER, VAL_FOLDER, TEST_FOLDER, LABELS_FOLDER):
    os.makedirs(base_path + TRAIN_DATA_AUG_FOLDER)
    os.makedirs(base_path + TRAIN_FOLDER)
    os.makedirs(base_path + VAL_FOLDER)
    os.makedirs(base_path + TEST_FOLDER)
    os.makedirs(base_path + LABELS_FOLDER)

def build_data_aug_folder(paths, labels, base_path, image_folder, labels_file_path):
    relative_paths = []
    new_labels = []
    for (img_path, label) in zip(paths, labels):
        file_name = get_file_name(img_path)
        relative_file_location = f"{image_folder}/{file_name}"
        
        with Image.open(img_path) as img:
            img = img.convert("L")
            for i in range(5):
                img_aug = np.array(img, dtype=np.float32) / 255.0
                img_aug = da.apply_all_techniques(img_aug)
                img_aug = np.clip(img_aug * 255, 0, 255).astype(np.uint8)
                
                img_aug = Image.fromarray(img_aug)
                img_aug_name = f"{relative_file_location}-{i}.png"
                img_aug.save(f"{base_path}{img_aug_name}", format="PNG")

                relative_paths.append(img_aug_name)
                new_labels.append(label)

                print(img_aug_name, label)

    df = pd.DataFrame({"path":relative_paths, "label":new_labels})
    df.to_csv(f"{base_path}{labels_file_path}", mode="a", header=False, index = False)

    
def build_split_folder(paths, labels, base_path, image_folder, labels_file_path):
    relative_paths = []
    for (img_path, label) in zip(paths, labels):
        file_name = get_file_name(img_path)
        file_name_png = f"{image_folder}/{file_name}.png"
        relative_paths.append(file_name_png)

        print(file_name_png, label)

        with Image.open(img_path) as img:
            img = img.convert("L")
            img.save(f"{base_path}{file_name_png}", format="PNG")

    df = pd.DataFrame({"path":relative_paths, "label":labels})
    df.to_csv(f"{base_path}{labels_file_path}", index = False)


def get_file_name(img_path):
    path_split = img_path.split('/')
    file_name = path_split[-1]
    file_name_no_extention = file_name.split('.')[0]

    return file_name_no_extention
