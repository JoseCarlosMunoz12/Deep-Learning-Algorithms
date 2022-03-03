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
        base_dir = os.getcwd() + '/data'
    if fileName == "":
        fName = input("Enter File Name:\n")
    else:
        fName = fileName
    file = os.path.join(base_dir, fName)
    try:
        data = pd.read_csv(file)
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
                        data = pd.read_csv(fName1)
                        rep = False
                    except:
                        print('incorrect file name')
    return data


def separate_data(data):
    print(data)
    cols = data.shape[1]
    rows = data.shape[0]
    X = np.array(data.iloc[:, 0:cols - 2], np.float32)
    y = np.array(data.iloc[:, cols - 2:cols - 1], np.float32)
    rng = default_rng()
    # select training and testing data
    train_idx = rng.choice(rows, size=math.ceil(rows * 3 / 4), replace=False)
    x_train = X[train_idx, :]
    y_train = y[train_idx, :]
    test_idx = np.setdiff1d(np.array(range(0, rows)), train_idx)
    x_test = X[test_idx, :]
    y_test = y[test_idx, :]
    return


def main():
    # Loading file
    data = get_file('house.csv')
    separate_data(data)
    # (x_test, x_pTrain, x_valid), (y_test, y_pTrain, y_valid) =
    exit()
    # Set results as categorical
    y_test = tf.keras.utils.to_categorical(y_test)
    y_pTrain = tf.keras.utils.to_categorical(y_pTrain)
    y_valid = tf.keras.utils.to_categorical(y_valid)
    x_dimen = len(x_test[0])
    # Create ANN network
    house_model = models.Sequential()
    house_model.add(layers.Dense(64, activation='relu', input_shape=(x_dimen,)))
    house_model.add(layers.Dense(32, activation='relu'))
    house_model.add(layers.Dense(16, activation='relu'))
    house_model.add(layers.Dense(8, activation='relu'))
    house_model.add(layers.Dense(4, activation='softmax'))
    house_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                        metrics=['accuracy'])
    # Fit Data
    history = house_model.fit(x_pTrain, y_pTrain,
                              epochs=45,
                              batch_size=512,
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
