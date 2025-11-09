from PIL import Image
import numpy as np
import cv2

import data_augmentation as da
import pandas as pd

def apply_all_techniques(image):

    image = random_gaussian_blur(image)
    image = random_dilate(image)
    image = random_erode(image)
    image = random_brightness(image)
    image = random_noise(image)

    return image


def random_gaussian_blur(image, probability=0.25):
    if np.random.rand() < probability:
        random_kernel_size = (generate_random_odd_number(), generate_random_odd_number())
        image = gaussian_blur(image, random_kernel_size)
    return image

def gaussian_blur(img, kernel_size):
    img = cv2.GaussianBlur(img, kernel_size, 0)
    return img

def random_invert(img):
        if np.random.rand() < 0.1:
            img = invert(img)
        return img

def invert(img):
    return 1.0 - img

def random_noise(img, probability = 0.25, max_noise_multiplier = 25):
    if np.random.rand() < probability:
        random_noise_multiplier = np.random.randint(1, max_noise_multiplier)
        img = noise(img, random_noise_multiplier)
    return img

def noise(img, noise_multiplier):
    noise_scale = noise_multiplier / 255.0 * 2
    noise = (np.random.random(img.shape) - 0.5) * noise_scale
    img = np.clip(img + noise, -1, 1)
    return img

def random_brightness(img, probability = 0.5, min_brightness = 0.25):
    if np.random.rand() < probability:
        random_brightness_factor = min_brightness + np.random.rand() * (1 - min_brightness)
        img = darken(img, random_brightness_factor)
    return img

def darken(img, brightness_factor):
    img = img * brightness_factor
    return img

def random_erode(img, probability = 0.25):
        if np.random.rand() < probability:
            img = erode(img)
        return img

def erode(img):
    kernel = np.ones((3, 3))
    img = cv2.erode(img, kernel)
    return img

def random_dilate(img, probability = 0.25):
    if np.random.rand() < probability:
        img = dilate(img)
    return img

def dilate(img):
    kernel = np.ones((3, 3))
    img = cv2.dilate(img, kernel)
    return img

def generate_random_odd_number():
    return np.random.randint(1, 4) * 2 + 1

def random_gaussian_blur(img, probability = 0.25):
    if np.random.rand() < probability:
        random_kernel_size = (generate_random_odd_number(), generate_random_odd_number())
        img = gaussian_blur(img, random_kernel_size)
    return img

def gaussian_blur(img, kernel_size):
    img = cv2.GaussianBlur(img, kernel_size, 0)
    return img


def build_data_aug_folder(base_path, source_labels_file, dest_folder, dest_labels_file):
    df = pd.read_csv(base_path + source_labels_file)

    new_image_paths = []
    new_labels = []

    for row in df.itertuples(index=False):

        img_path = base_path + row.path
        with Image.open(img_path) as img:
            img = img.convert("L")

            for i in range(5):
                img_aug = np.array(img, dtype=np.float32) / 255.0
                img_aug = da.apply_all_techniques(img_aug)
                img_aug = np.clip(img_aug * 255, 0, 255).astype(np.uint8)

                img_aug = Image.fromarray(img_aug)
                file_name = row.path.split("/")[-1].split(".")[0]
                relative_path_img_aug = f"{dest_folder}/{file_name}-{i}.png"
                img_aug.save(f"{base_path}{relative_path_img_aug}", format="PNG")

                new_image_paths.append(relative_path_img_aug)
                new_labels.append(row.label)

                print(relative_path_img_aug, row.label)

    dest_df = pd.DataFrame({"path":new_image_paths, "label":new_labels})
    dest_df.to_csv(f"{base_path}{dest_labels_file}", mode="a", header=False, index=False)