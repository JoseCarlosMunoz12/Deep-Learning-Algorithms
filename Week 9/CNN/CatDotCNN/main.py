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


def load_images():
    if not IN_COLAB:
        original_Dataset_Dir_dogs = os.getcwd() + "\\dogs"
        original_Dataset_Dir_cats = os.getcwd() + "\\cats"
    else:
        drive.mount()
        original_Dataset_Dir_dogs = os.getcwd() + "\\dogs"
        original_Dataset_Dir_cats = os.getcwd() + "\\cats"

    return {original_Dataset_Dir_dogs, original_Dataset_Dir_cats}


def parse_classes():
    pass


def main():
    print('test')


if __name__ == '__main__':
    main()
