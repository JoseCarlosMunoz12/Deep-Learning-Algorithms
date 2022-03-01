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


def main():
    # Loading file
    file = get_file('house.txt')
    lines = file.readlines()
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
    train_idx = rng.choice(rows, size=math.ceil(rows*3/4), replace=False)
    x_train = []
    y_train = []
    for train in train_idx:
        x_train.append(data[train])
        y_train.append(labels[train])
    test_idx = np.setdiff1d(np.array(range(0, rows)), train_idx)
    x_test = []
    y_test = []
    for test in test_idx:
        x_train.append(data[test])
        y_train.append(labels[test])



if __name__ == '__main__':
    main()
