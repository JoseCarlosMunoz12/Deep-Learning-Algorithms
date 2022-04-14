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


def normalize_reformat_prepare(dataset, n_features, n_hours, n_in=1, n_out=1):
    values = dataset.values
    encoder = LabelEncoder()
    values[:, 10] = encoder.fit_transform(values[:, 10])
    values[:, 12] = encoder.fit_transform(values[:, 12])
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, n_in, n_out)
    cols = []
    n_obs = n_hours * n_features
    for j in range( n_obs, n_obs + n_features):
        cols.append(j)
    reframed.drop(reframed.columns[cols], axis=1, inplace=True)
    print(reframed.head())
    return reframed, scaler


def train_model(dataset, title, n_features, n_hours, n_in=1, n_out=1):
    reframed, scaler = normalize_reformat_prepare(dataset, n_features, n_hours, n_in, n_out)
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
    print(train_x.shape[0])
    print(train_x.shape[1])
    print(train_x.shape[2])
    model = models.Sequential()
    model.add(layers.LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(layers.Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit network
    history = model.fit(train_x, train_y, epochs=50, batch_size=96, validation_data=(valid_x, valid_y), verbose=2,
                        shuffle=False)
    plot_results(history)
    # seeing results of the data
    y_hat = model.predict(test_x)
    rmse0 = sqrt(mean_squared_error(test_y,y_hat))
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    # invert scaling for forecast
    inv_y_hat = concatenate((y_hat, test_x[:, 1:]), axis=1)
    inv_y_hat = scaler.inverse_transform(inv_y_hat)
    inv_y_hat = inv_y_hat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    rmse = sqrt(mean_squared_error(inv_y, inv_y_hat))
    print('Test RMSE scaled: %.3f' % rmse0)
    print('Test RMSE absolute: %.3f' % rmse)


def main():
    # Loading file
    dataset = loadfile()
    train_model(dataset, 'Two Day Prediction', 12, 3, 1, 1)


if __name__ == '__main__':
    main()
