import tensorflow as tf
import numpy as np
import math
import os
from numpy.random import default_rng

from Neuron_Class import Neuron

# A = tf.Variable([[3, 2], [5, 2]])
# B = tf.constant([[9, 5], [1, 3]])
# AB_con = tf.concat(values=[A, B], axis=1)
# print("Adding B\'s columns to A:\n", AB_con)

# x = tf.Variable(3.0)
# with tf.GradientTape() as tape:
#    y = x ** 2
# dy_dx = tape.gradient(y, x)
# print(dy_dx)

# linear predictor here

# w = tf.Variable([[-1.25, 0.56],
#                 [-0.62, -0.39],
#                 [1.82, -0.25]])
# b = tf.Variable(tf.zeros(2, dtype=tf.float32, name='b'))
# x = [[1.0, 2.0, 3.0]]
# with tf.GradientTape(persistent=True) as tape:
#    y = x @ w + b
#    loss = tf.reduce_mean(y**2)
# [dl_dw, dl_db] = tape.gradient(loss, [w, b])
# print(w)
# print(b)
# print('x=', x)
# print('y=', y)
# print('l=', loss)
# print('dw=', dl_dw)
# print('db=', dl_db)
# print('shapes:')
# print(w.shape)
# print(dl_dw.shape)

# functions in tensor flow

# W = tf.Variable(tf.ones(shape=[2, 2]), name='W')
# b = tf.Variable(tf.ones(shape=2), name='b')


# @tf.function
# def forward(x):
#    s = tf.linalg.matvec(W, x) + b
#    return s


# out_a = forward(tf.constant([1.0, 1.0], shape=2))

base_dir = os.getcwd() + '\\data'

fName = input('Enter a filename (must be in the standard working folder')
file = os.path.join(base_dir, fName)
try:
    data = open(file, 'r')
    ferror = True
except FileNotFoundError:
    ferror = False
if ferror:
    rep = True
    while rep:
        fName1 = 'q'
        if fName1 == 'q':
            exit()
        else:
            try:
                data = open(fName1, 'r')
                rep = False
            except:
                print('incorrect file name')
