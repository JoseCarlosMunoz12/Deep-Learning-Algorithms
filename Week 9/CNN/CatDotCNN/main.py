import tensorflow as tf
from keras.utils import np_utils
from keras import models
from keras import layers
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator
import shutil
import numpy as np
import os

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
    else:
        drive.mount()
        original_dataset_dir_dogs = os.getcwd() + "\\Images\\PetImages\\Dog"
        original_dataset_dir_cats = os.getcwd() + "\\Images\\PetImages\\Cat"
    return original_dataset_dir_dogs, original_dataset_dir_cats


def check_paths(dog_paths, cat_paths):

    if not os.path.exists(dog_paths):
        print('no dog data directory')
    if not os.path.exists(cat_paths):
        print('no cat data directory')
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
    # Parse images into each folder
    dogs_dirs = [train_dogs_dir, validation_dogs_dir, test_dogs_dir]
    cats_dirs = [train_cats_dir,  validation_cats_dir,  test_cats_dir]
    train_test_val = [train_dir, test_dir, validation_dir]
    return dogs_dirs, cats_dirs, train_test_val


def parse_files(dog_paths, cat_paths, og_dog, og_cat):
    ranges = [[0, 400], [400, 600], [600, 1000]]
    for i in range(3):
        limits = ranges[i]
        if len(os.listdir(dog_paths[i])) == 0:
            fnames = ['{}.jpg'.format(j) for j in range(limits[0], limits[1])]
            for fname in fnames:
                src = os.path.join(og_dog, fname)
                dst = os.path.join(dog_paths[i], fname)
                shutil.copyfile(src, dst)
        if len(os.listdir(cat_paths[i])) == 0:
            fnames = ['{}.jpg'.format(j) for j in range(limits[0], limits[1])]
            for fname in fnames:
                src = os.path.join(og_cat, fname)
                dst = os.path.join(cat_paths[i], fname)
                shutil.copyfile(src, dst)


def prepare_data(paths):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        paths[0], target_size=(150, 150), batch_size=20, class_mode='binary')
    validation_generator = datagen.flow_from_directory(
        paths[2], target_size=(150, 150), batch_size=20, class_mode='binary')
    test_generator = datagen.flow_from_directory(
        paths[1], target_size=(150, 150), batch_size=20, class_mode='binary')


def main():
    dogs_paths, cats_paths = images_paths()
    base_dir = check_paths(dogs_paths, cats_paths)
    dog_paths, cat_paths, ttv_dirs = make_dirs(base_dir)
    parse_files(dog_paths, cat_paths, dogs_paths, cats_paths)


if __name__ == '__main__':
    main()
