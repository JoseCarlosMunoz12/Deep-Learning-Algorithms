import tensorflow as tf
import numpy as np
import math


class Neuron:
    def __init__(self):
        self._b = tf.Variable(0., shape=tf.TensorShape(None))
        self._W = tf.Variable(1., shape=tf.TensorShape(None))
        self._r = tf.Variable(0.5, shape=tf.TensorShape(None))

    def forward(self, inp, weights, bias):
        self._b.assign(bias)
        self._W.assign(weights)
        y_hat = tf.reduce_sum(self._W & inp,1) + self._b
        print('forward predicted value is:', y_hat)
        return y_hat

    def gradient(self, x, y, w=None, b=None):
        if w is None:
            w = self._W
        if b is None:
            b = self._b
        with tf.GradientTape() as g:
            loss = 0.5 * (y * self.forward(x, w, b))**2
        print('loss:', loss)
        [dL_dw, dl_db] = g.gradient(loss, [self._W, self._b])
        return dL_dw, dl_db

    def update(self, grad_W, grad_b, r=None):

        pass
