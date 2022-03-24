import tensorflow as tf
from keras.utils import np_utils
import matplotlib.pyplot as plt
import shutil
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers

try:
    from google.colab import drive
    IN_COLAB = True
    from keras.applications.vgg16 import VGG16
except:
    IN_COLAB = False
    from keras.applications import VGG16


def images_paths():
    if not IN_COLAB:
        original_dataset_dir_dogs = os.getcwd() + "\\Images\\PetImages\\Dog"
        original_dataset_dir_cats = os.getcwd() + "\\Images\\PetImages\\Cat"
        original_dataset_dir_elephants = os.getcwd() + "\\Images\\PetImages\\Elephants"
    else:
        drive.mount()
        original_dataset_dir_dogs = os.getcwd() + "\\Images\\PetImages\\Dog"
        original_dataset_dir_cats = os.getcwd() + "\\Images\\PetImages\\Cat"
        original_dataset_dir_elephants = os.getcwd() + "\\Images\\PetImages\\Elephants"
    return original_dataset_dir_dogs, original_dataset_dir_cats,  original_dataset_dir_elephants


def check_paths(dog_paths, cat_paths, elephant_paths):
    if not os.path.exists(dog_paths):
        print('no dog data directory')
    if not os.path.exists(cat_paths):
        print('no cat data directory')
    if not os.path.exists(elephant_paths):
        print('no elephant data directory')
    if not IN_COLAB:
        base_dir = os.getcwd() + '\\Images\\cats_and_dogs_small'
    else:
        base_dir = '/content/gdrive/My Drive/courses/Deep Learning/data/cats_and_dogs_small'
    if not os.path.exists(base_dir):
        print('creating new directory of images')
        os.mkdir(base_dir)
    return base_dir


def make_dirs(base_dir):
    # Make Training folder
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    train_cats_dir = os.path.join(train_dir, 'cats')
    if not os.path.exists(train_cats_dir):
        os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    if not os.path.exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)
    train_elephants_dir = os.path.join(train_dir, 'elephants')
    if not os.path.exists(train_elephants_dir):
        os.mkdir(train_elephants_dir)
    # Make validation folder
    validation_dir = os.path.join(base_dir, 'validation')
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    if not os.path.exists(validation_cats_dir):
        os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    if not os.path.exists(validation_dogs_dir):
        os.mkdir(validation_dogs_dir)
    validation_elephants_dir = os.path.join(validation_dir, 'elephants')
    if not os.path.exists(validation_elephants_dir):
        os.mkdir(validation_elephants_dir)
    # Make Test folder
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    test_cats_dir = os.path.join(test_dir, 'cats')
    if not os.path.exists(test_cats_dir):
        os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    if not os.path.exists(test_dogs_dir):
        os.mkdir(test_dogs_dir)
    test_elephants_dir = os.path.join(test_dir, 'elephants')
    if not os.path.exists(test_elephants_dir):
        os.mkdir(test_elephants_dir)
    # Parse images into each folder
    dogs_dirs = [train_dogs_dir, validation_dogs_dir, test_dogs_dir]
    cats_dirs = [train_cats_dir,  validation_cats_dir,  test_cats_dir]
    elephants_dirs = [train_elephants_dir, validation_elephants_dir, test_elephants_dir]
    train_test_val = [train_dir, test_dir, validation_dir]
    return dogs_dirs, cats_dirs, elephants_dirs, train_test_val


def main():
    dogs_paths, cats_paths, elephants_paths = images_paths()
    base_dir = check_paths(dogs_paths, cats_paths, elephants_paths)
    dog_paths, cat_paths, elephant_paths, ttv_dirs = make_dirs(base_dir)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
