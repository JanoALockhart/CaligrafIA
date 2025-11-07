import tensorflow as tf
import numpy as np
import data_augmentation as da
import settings

@tf.py_function(Tout=tf.float32)
def apply_augmentations(image):

    if settings.DEBUG_MODE:
        print("Before data aug:", image.shape)

    image = image.numpy()
    image = np.squeeze(image)
    image = da.random_gaussian_blur(image)
    image = da.random_dilate(image)
    image = da.random_erode(image)
    image = da.random_brightness(image)
    image = da.random_noise(image)
    image = np.expand_dims(image, axis=-1)
    if settings.DEBUG_MODE:
        print("After data aug image shape:", image.shape)

    return image
