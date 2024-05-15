"""
Module used to load Mechanic Components dataet
https://www.kaggle.com/datasets/satishpaladi11/mechanic-component-images-normal-defected/data
"""

import os

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split
from utils.training_utils import DatasetConfig

def get_label(file_path: str, class_names: list[str], num_classes: int) -> list[int]:
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    
    n = class_names.index(parts[-2])
    return list(map(lambda i: 1 if i == n else 0, range(num_classes))) 

def decode_img(img: tf.Tensor, image_size: list[int]) -> tf.Tensor:
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # resize the image to the desired size.
    return tf.image.resize(img, image_size)

def process_path(file_path: str, image_size: list[int], class_names: list[str], num_classes: int):
    """Creates training entry by processing the path and its contents"""

    label = get_label(file_path, class_names, num_classes)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, image_size)
    
    return img, label

def load(batch_size: int, seed: int) -> DatasetConfig:
    base_path = 'data/MechanicComponents'

    filenames = tf.io.gfile.glob(str(base_path + '/Defected1/*'))
    filenames.extend(tf.io.gfile.glob(str(base_path + '/Defected2/*')))
    filenames.extend(tf.io.gfile.glob(str(base_path + '/Normal/*')))

    train_filenames, test_filenames = train_test_split(filenames, test_size=0.2)

    class_names = np.array([str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]
                        for item in tf.io.gfile.glob(str(base_path + "/*"))])
    class_names = list(class_names)

    num_classes = len(class_names)
    image_size, image_shape = (80, 80), (80, 80, 3)

    train = zip(*list(map(lambda f: process_path(f, image_size, class_names, num_classes), train_filenames)))
    train_data, train_input = [list(t) for t in train]
    # train_integer_size = len(train_data) // batch_size
    train_data = np.array(train_data) #[:train_integer_size]
    train_input = np.array(train_input) #[:train_integer_size]

    test = zip(*list(map(lambda f: process_path(f, image_size, class_names, num_classes), test_filenames)))
    test_data, test_input = [list(t) for t in test]
    # test_integer_size = len(test_data) // batch_size
    test_data = np.array(test_data) #[:test_integer_size]
    test_input = np.array(test_input) #[:test_integer_size]

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        # horizontal_flip = True,
        zoom_range = 0.2,
        shear_range = 0.2,
        rotation_range=40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        validation_split = 0.2,
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    train_ds = train_generator.flow(train_data, train_input, subset='training', batch_size=batch_size, shuffle=True, seed=seed)
    val_ds = train_generator.flow(train_data, train_input, subset='validation', batch_size=batch_size, shuffle=True, seed=seed)
    test_ds = test_generator.flow(test_data, test_input, batch_size=batch_size, shuffle=True, seed=seed)
    datasets = [train_ds, val_ds, test_ds]

    return DatasetConfig(image_shape, num_classes, datasets, steps_per_epoch=len(train_ds), validation_steps=len(val_ds))

    