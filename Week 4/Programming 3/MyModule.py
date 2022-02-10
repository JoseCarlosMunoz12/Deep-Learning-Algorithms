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
