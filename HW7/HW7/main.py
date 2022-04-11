import tensorflow as tf
from pandas import read_csv
from datetime import datetime
import os
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import concat
from keras import models
from keras import layers
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error
from numpy import concatenate

try:
    from google.colab import drive

    IN_COLAB = True
except:
    IN_COLAB = False


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


def loadfile():
    if not IN_COLAB:
        dir_path = os.getcwd()
        data_dir = os.path.join(dir_path, 'pollution')
    else:
        drive.mount('/content/gdrive')
        dir_path = os.path.dirname(os.path.realpath(os.path.abspath('')))
        data_dir = os.path.join(dir_path, '/content/gdrive/My Drive/courses/Deep Learning/data/pollution')
    if not os.path.exists(data_dir):
        exit(3)
    dst = os.path.join(data_dir, 'pollution.csv')
    copy = False
    if not os.path.exists(dst):
        print('copying file and reformatting')
        copy = True
        src = os.path.join(data_dir, 'pollution_for_students.csv')
        dataset = read_csv(src, parse_dates=[['year', 'month', 'day', 'hour']],
                           index_col=0, date_parser=parse)
        dataset.drop('No', axis=1, inplace=True)
        dataset.columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
                           'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM',
                           'station']
        dataset.index.name = 'date'
        dataset['PM2.5'].fillna(0, inplace=True)
        dataset = dataset[24:]
        dataset.to_csv(dst)
    if not copy:
        print('reading pollution file')
        dataset = read_csv(dst, header=0, index_col=0)
    return dataset


def main():
    # Loading file
    dataset = loadfile()
    print(dataset)


if __name__ == '__main__':
    main()
