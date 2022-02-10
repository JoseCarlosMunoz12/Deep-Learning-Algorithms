import tensorflow as tf
from tensorflow import keras
import numpy as np


class MyModel(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self._w_variable = tf.Variable(1., shape=tf.TensorShape(None))
        self._b_variable = tf.Variable(0., shape=tf.TensorShape(None))
        self._first_run = tf.Variable(0, shape=[], trainable=False)

    def __call__(self, x, w=None, b=None):
        if w is not None and self._first_run == 0:
            if len(w.shape) == 1:
                w = tf.reshape(x, [w.shape[0], 1])
            self._first_run.assign(1)
            self._w_variable.assign(w)
            if b is not None:
                self._b_variable.assign(b)
        return tf.matmul(self._w_variable, x) + self._b_variable


class NewModel(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self._layer_1 = MyModel()
        self._layer_2 = MyModel()

    def __call__(self, x, w1=None, b1=None, w2=None, b2=None):
        if w1 is None:
            y_1 = self._layer_2(x, w2, b2)
            return y_1
        else:
            y_2 = self._layer_1(x, w1, b1)
            y_2 = self._layer_2(y_2, w2, b2)
            return y_2


class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init()
        w_init = tf.random_normal_initializer()
        self.__w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True
        )
        b_init = tf.zeros_initializer()
        self.__b = tf.Variable(
            initial_value=w_init(shape=(units,), dtype="float32"),
            trainable=True
        )

    def __call__(self, inputs):
        return tf.matmul(inputs, self.__w) + self.__b
