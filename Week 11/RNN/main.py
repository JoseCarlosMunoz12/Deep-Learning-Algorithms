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

try:
    from google.colab import drive
    IN_COLAB=True
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
        src = os.path.join(data_dir, 'raw_pollution.csv')
        dataset = read_csv(src, parse_dates=[['year', 'month', 'day', 'hour']],
                           index_col=0, date_parser=parse)
        dataset.drop('No', axis=1, inplace=True)
        dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
        dataset.index.name = 'date'
        dataset['pollution'].fillna(0, inplace=True)
        dataset = dataset[24:]
        dataset.to_csv(dst)
    if not copy:
        print('reading pollution file')
        dataset = read_csv(dst, header=0, index_col=0)
    return dataset


def plot_data(dataset):
    values = dataset.values
    groups = list(range(0, 7+1))
    i = 1
    plt.figure()
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])
        plt .title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def normalize_reformat_prepare(dataset):
    values = dataset.values
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    print(reframed.head())
    return reframed


def plot_results(history):
    plt.close()
    plt.plot(history.history['mse'], 'r',label='MSE')
    plt.plot(history.history['val_mse'], 'b', label='Validation MSE')
    plt.title('Training and validation Mean Squared Error')
    plt.legend()
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.title('Training and validation Loss (MAE)')
    plt.legend()
    plt.show()


def main():
    dataset = loadfile()
    plot_data(dataset)
    reframed = normalize_reformat_prepare(dataset)
    # prepare data
    values = reframed.values
    n_train_hours = int(round(len(values)/3))
    n_valid_hours = int(round(2 * len(values)/3))
    train = values[: n_train_hours, :]
    valid = values[n_train_hours:n_valid_hours, :]
    test = values[n_valid_hours:, :]
    # split dat into training input, validation output and later testing input
    train_x, train_y = train[:, :-1], train[:, -1]
    valid_x, valid_y = valid[:, :-1], valid[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D[samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    valid_x = valid_x.reshape((valid_x.shape[0], 1, valid_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    # Display shape of models
    print(train_x.shape, train_y.shape)
    print(valid_x.shape, valid_y.shape)
    print(test_x.shape, test_y.shape)
    # make model
    model = models.Sequential()
    print(train_x.shape[0])
    print(train_x.shape[1])
    print(train_x.shape[2])
    model.add(layers.LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(layers.Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit network
    history = model.fit(train_x, train_y, epochs=50, batch_size=96, validation_data=(valid_x, valid_y), verbose=2,
                        shuffle=False)
    plot_results(history)


if __name__ == '__main__':
    main()
