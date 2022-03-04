import tensorflow as tf
from keras import models
from keras import layers
import numpy as np
import math
import os
import pandas as pd
from numpy.random import default_rng
import matplotlib.pyplot as plt

try:
    from google.colab import drive

    IN_COLAB = True
except:
    IN_COLAB = False


# function for converting string to float
def mk_float(s):
    s = s.strip()
    return float(s) if s else 0.0


def get_file(fileName=""):
    if IN_COLAB:
        drive.mount('/content/gdrive')
        base_dir = '/content/gdrive/My Drive/courses/Deep Learning/data'
        if not os.path.exits(base_dir):
            os.mkdir(base_dir)
    else:
        base_dir = os.getcwd() + '\\data'
    if fileName == "":
        fName = input("Enter File Name:\n")
    else:
        fName = fileName
    file = os.path.join(base_dir, fName)
    try:
        data = pd.read_csv(file, header=None, names=['0','1','2','3',\
                                             '4','5','6','7',\
                                             '8','ocean_proximity'])
        ferror = False
    except FileNotFoundError:
        ferror = True
    if IN_COLAB and ferror:
        from google.colab import files
        files.upload()
    else:
        if ferror:
            rep = True
            while rep:
                fName1 = input("Enter complete path to the data or q (to quit):")
                if fName1 == 'q':
                    exit
                else:
                    try:
                        data = pd.read_csv(fName1, header=None,  names=['longitude','latitude','housing_median_age','total_rooms',\
                                             'total_bedrooms','population','households','median_income',\
                                             'median_house_value','ocean_proximity'])
                        rep = False
                    except:
                        print('incorrect file name')
    return data


def separate_data(data):
    # set values of categorical column
    data = data.fillna(0)
    cleanup_nums = {"ocean_proximity": {"INLAND": 0, "<1H OCEAN": 1, "NEAR OCEAN": 2, "NEAR BAY": 3, "ISLAND": 4}}
    data = data.replace(cleanup_nums)
    cols = data.shape[1]
    rows = data.shape[0]
    x = np.array(data.iloc[:, 0:-2], np.float32)
    x = x.astype('float32')
    x = (x - x.min(axis=0)) / x.ptp(axis=0)
    y = np.array(data.iloc[:, -1], np.int32)
    y = tf.keras.utils.to_categorical(y)
    rng = default_rng()
    # select training and testing data
    train_idx = rng.choice(rows, size=math.ceil(rows * 3 / 4), replace=False)
    x_train = x[train_idx, :]
    y_train = y[train_idx]
    test_idx = np.setdiff1d(np.array(range(0, rows)), train_idx)
    x_test = x[test_idx, :]
    y_test = y[test_idx]
    # select training and validation data from the training set
    rows = x_train.shape[0]
    part_train = rng.choice(rows, size=math.ceil(rows * 3 / 4), replace=False)
    x_p_train = x_train[part_train, :]
    y_p_train = y_train[part_train]
    valid_train = np.setdiff1d(np.array(range(0, rows)), part_train)
    x_valid = x_train[valid_train, :]
    y_valid = y_train[valid_train]
    return(x_test, x_p_train, x_valid), (y_test, y_p_train, y_valid)


def main():
    # Loading file
    data = get_file('house.csv')
    (x_test, x_pTrain, x_valid), (y_test, y_pTrain, y_valid) = separate_data(data)
    # Set results as categorical
    x_dimen = len(x_test[0])
    # Create ANN network
    house_model = models.Sequential()
    house_model.add(layers.Dense(64, activation='relu', input_shape=(x_dimen, )))
    house_model.add(layers.Dense(32, activation='relu'))
    house_model.add(layers.Dense(16, activation='relu'))
    house_model.add(layers.Dense(8, activation='relu'))
    house_model.add(layers.Dense(5, activation='softmax'))
    house_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                        metrics=['accuracy'])
    # Fit Data
    history = house_model.fit(x_pTrain, y_pTrain,
                              epochs=32,
                              batch_size=128,
                              validation_data=(x_valid, y_valid))
    # Plotting data of the training and validation loss
    history_dict = history.history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # Plotting data of the training and validation Accuracy
    plt.clf()
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'g^', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # Evaluate the Data Set
    l, acc = house_model.evaluate(x_test, y_test)
    print('accuracy is: ', acc)
    print(house_model.predict(x_test))


if __name__ == '__main__':
    main()
