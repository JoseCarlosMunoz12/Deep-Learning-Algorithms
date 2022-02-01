import tensorflow as tf
import numpy as np
import keras_preprocessing as kr

d = tf.zeros((2, 2))
e = tf.ones((2, 2))
print(d)
print(e)

a1 = tf.constant(5)
b1 = tf.constant(2)
c1 = tf.add(a1, b1, name='Add')
print(c1)

print(d)
print(tf.reshape(d, (1, 4)))
a1 = a1.numpy()
a2 = d.numpy()
print(a1, '\n', a2)

tens_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=float)
print(tens_3d[1, 1, 0])

real = tf.constant([2.25, 3.25])
imag = tf.constant([4.75, 5.75])
y = tf.complex(real, imag)
print(y)

b = tf.constant([1, 2])
Al = tf.constant([[2, 24], [2, 26], [2, 57]])
x = [[1, 2, 3]]
Cl = tf.matmul(x, Al) + b
print(Cl)

y = x @ Al + b
print(y)

x = tf.constant([[1], [2], [3]])
print('x=', x)
y = tf.range(1, 5)
print('y=', y)
c = x * y
print('product c=', c)

