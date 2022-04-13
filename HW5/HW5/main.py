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
    from keras.applications.inception_v3 import InceptionV3
except:
    IN_COLAB = False
    from keras.applications import VGG16
    from keras.applications import InceptionV3


def images_paths():
    if not IN_COLAB:
        original_dataset_dir_dogs = os.getcwd() + "\\Images\\PetImages\\Dog"
        original_dataset_dir_cats = os.getcwd() + "\\Images\\PetImages\\Cat"
        original_dataset_dir_elephants = os.getcwd() + "\\Images\\PetImages\\Elephant"
    else:
        drive.mount()
        original_dataset_dir_dogs = os.getcwd() + "\\Images\\PetImages\\Dog"
        original_dataset_dir_cats = os.getcwd() + "\\Images\\PetImages\\Cat"
        original_dataset_dir_elephants = os.getcwd() + "\\Images\\PetImages\\Elephant"
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


def parse_files(dog_paths, cat_paths, elephant_paths, og_dog, og_cat, og_elephant):
    ranges = [[1, 601], [601, 801], [801, 1201]]
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
        if len(os.listdir(elephant_paths[i])) == 0:
            fnames = ['{}.jpg'.format(j) for j in range(limits[0], limits[1])]
            for fname in fnames:
                src = os.path.join(og_elephant, fname)
                dst = os.path.join(elephant_paths[i], fname)
                shutil.copyfile(src, dst)


def prepare_data(paths):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        paths[0], target_size=(150, 150), batch_size=20, class_mode="categorical")
    validation_generator = datagen.flow_from_directory(
        paths[2], target_size=(150, 150), batch_size=20, class_mode="categorical")
    test_generator = datagen.flow_from_directory(
        paths[1], target_size=(150, 150), batch_size=20, class_mode="categorical")
    return train_generator, validation_generator, test_generator


# End to End training decision model with frozen dgg16 weights
def end_to_end():
    dog_paths, cat_paths, elephant_paths = images_paths()
    base_dir = check_paths(dog_paths, cat_paths, elephant_paths)
    dogs_paths, cats_paths, elephants_paths, ttv_dirs = make_dirs(base_dir)
    parse_files(dogs_paths, cats_paths, elephants_paths, dog_paths, cat_paths, elephant_paths)
    train_g, valid_g, test_g = prepare_data(ttv_dirs)
    # Preparing model
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    conv = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    op = tf.keras.optimizers.RMSprop(lr=2e-5)
    model = models.Sequential()
    model.add(conv)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    conv.trainable = False
    model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['acc'])
    # Display results
    history = model.fit(train_g, steps_per_epoch=20, epochs=7,
                        validation_data=valid_g, validation_steps=10, verbose=2)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training Acc')
    plt.plot(epochs, val_acc, 'b', label='Validation_Acc')
    plt.title('Training Validation Accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training Validation Loss')
    plt.legend()

    plt.show()
    plt.clf()

    test_loss, test_acc = model.evaluate(test_g, steps=20)
    print('End to End Results:')
    print('test acc:', test_acc)
    print('test loss:', test_loss)


# Deciding after preprocessing with pretrained NN
def extract_features(conv_base, directory, sample_count, batch_size=20):
    datagen = ImageDataGenerator(rescale=1./255)
    features = np.zeros(shape=(sample_count, 3, 3, 2048))
    labels = np.zeros(shape=(sample_count, 3))
    generator = datagen.flow_from_directory(directory, target_size=(150, 150),
                                            batch_size=batch_size, class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


def pre_processing():
    dog_paths, cat_paths, elephant_paths = images_paths()
    base_dir = check_paths(dog_paths, cat_paths, elephant_paths)
    dogs_paths, cats_paths, elephants_paths, ttv_dirs = make_dirs(base_dir)
    parse_files(dogs_paths, cats_paths, elephants_paths, dog_paths, cat_paths, elephant_paths)
    conv = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    op = tf.keras.optimizers.RMSprop(lr=2e-5)
    train_feat, train_labels = extract_features(conv, ttv_dirs[0], 600)
    valid_feat, valid_labels = extract_features(conv, ttv_dirs[2], 200)
    test_feat, test_labels = extract_features(conv, ttv_dirs[1], 400)
    train_feat = np.reshape(train_feat, (600, 3 * 3 * 2048))
    valid_feat = np.reshape(valid_feat, (200, 3 * 3 * 2048))
    test_feat = np.reshape(test_feat, (400, 3 * 3 * 2048))
    decision_model = models.Sequential()
    decision_model.add(layers.Dense(256, activation='relu', input_dim=3 * 3 * 2048))
    decision_model.add(layers.Dropout(0.5))
    decision_model.add(layers.Dense(3, activation='softmax'))

    decision_model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['acc'])

    decision_history = decision_model.fit(train_feat, train_labels,
                                          epochs=30,
                                          batch_size=20,
                                          validation_data=(valid_feat, valid_labels))

    d_acc = decision_history.history['acc']
    d_val_acc = decision_history.history['val_acc']
    d_loss = decision_history.history['loss']
    d_val_loss = decision_history.history['val_loss']

    d_epochs = range(len(d_acc))

    plt.plot(d_epochs, d_acc, 'bo', label='Training acc')
    plt.plot(d_epochs, d_val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(d_epochs, d_loss, 'bo', label='Training loss')
    plt.plot(d_epochs, d_val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    plt.clf()
    test_loss, test_acc = decision_model.evaluate(test_feat, test_labels)
    print('Pre Processing Results:')
    print('test acc:', test_acc)
    print('test loss:', test_loss)


# True End to End decision that includes training convolutional Layers of VGG16

def with_convolutional_training(set_trainable=False):
    dog_paths, cat_paths, elephant_paths = images_paths()
    base_dir = check_paths(dog_paths, cat_paths, elephant_paths)
    dogs_paths, cats_paths, elephants_paths, ttv_dirs = make_dirs(base_dir)
    parse_files(dogs_paths, cats_paths, elephants_paths, dog_paths, cat_paths, elephant_paths)
    train_g, valid_g, test_g = prepare_data(ttv_dirs)
    # Preparing model
    conv = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    op = tf.keras.optimizers.RMSprop(lr=2e-5)
    e_to_e_model = models.Sequential()
    e_to_e_model.add(conv)
    e_to_e_model.add(layers.Flatten())
    e_to_e_model.add(layers.Dense(256, activation='relu'))
    e_to_e_model.add(layers.Dense(3, activation='softmax'))
    # set parts to be trainable
    conv.trainable = True
    for layer in conv.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    e_to_e_model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['acc'])
    # Train AI
    history = e_to_e_model.fit(train_g, steps_per_epoch=20, epochs=10,
                               validation_data=valid_g, validation_steps=10,
                               verbose=2)
    e_acc = history.history['acc']
    e_val_acc = history.history['val_acc']
    e_loss = history.history['loss']
    e_val_loss = history.history['val_loss']

    e_epochs = range(len(e_acc))

    plt.plot(e_epochs, e_acc, 'bo', label='Training acc')
    plt.plot(e_epochs, e_val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(e_epochs, e_loss, 'bo', label='Training loss')
    plt.plot(e_epochs, e_val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    plt.clf()
    e_test_loss, e_test_acc = e_to_e_model.evaluate(test_g, steps=20)
    print('Convolutional Training Results:')
    print('test acc:', e_test_acc)
    print('test loss:', e_test_loss)


# Run items
def main():
    end_to_end()
    pre_processing()
    with_convolutional_training()
    # From all of these methods, the method with convolutional training performed the best.
    # This is because some of the weights and blocks in the CNN are also being trained.
    # This leads to higher accuracy but takes much more time to complete


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
