import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import math
import os
from numpy.random import default_rng

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
        data = open(file, 'r')
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
                        data = open(fName1, 'r')
                        rep = False
                    except:
                        print('incorrect file name')
    return data


def separate_data(lines):
    dic = {}
    data = []  # holds all of feature for each point
    labels = []  # holds all of the labels for each point
    # Parsing data
    for line in lines:
        newstr = line.strip()
        parsed = newstr.split(',')
        if not (parsed[-1]) in dic:
            dic[parsed[-1]] = len(dic)
        temp = []
        for item in parsed[:-1]:
            temp.append(mk_float(item))
        data.append(temp)
        labels.append(dic[parsed[-1]])
    # Seperate dataand labels to training, validation and testing set
    cols = len(data[1])
    rows = len(data)
    rng = default_rng()
    # Select training and testing data
    train_idx = rng.choice(rows, size=math.ceil(rows * 3 / 4), replace=False)
    test_idx = np.setdiff1d(np.array(range(0, rows)), train_idx)
    x_test = []
    y_test = []
    for test in test_idx:
        x_test.append(data[test])
        y_test.append(labels[test])
    x_train = []
    y_train = []
    for train in train_idx:
        x_train.append(data[train])
        y_train.append(labels[train])
    # separate training set to partial training and validation set
    partial_test_set = rng.choice(len(x_train), size=math.ceil((len(x_train)) * 8 / 10), replace=False)
    valid_set = np.setdiff1d(np.array(range(0, len(x_train))), partial_test_set)
    x_p_train = []
    y_p_train = []
    for p_train in partial_test_set:
        x_p_train.append(data[p_train])
        y_p_train.append((labels[p_train]))
    x_validation = []
    y_validation = []
    for validation in valid_set:
        x_validation.append(data[validation])
        y_validation.append(labels[validation])
    return (x_test, x_p_train, x_validation), (y_test, y_p_train, y_validation)


def main():
    # Loading file
    file = get_file('house.txt')
    lines = file.readlines()
    (x_test, x_pTrain, x_valid), (y_test, y_pTrain, y_valid) = separate_data(lines)
    print(len(x_test))
    print(len(x_pTrain))
    print(len(x_valid))
    print(len(y_test))
    print(len(y_pTrain))
    print(len(y_valid))


if __name__ == '__main__':
    main()
