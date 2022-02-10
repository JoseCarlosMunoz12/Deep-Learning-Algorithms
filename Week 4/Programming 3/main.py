import tensorflow as tf
from tensorflow import keras
import numpy as np


from MyModule import MyModel as Mm
from MyModule import Linear
model = Mm()
w_0 = tf.constant([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
b_0 = tf.constant( [[1.0], [2.0], [1.0]])
x = [[1.0], [1.0]]
p = model(x, w_0, b_0).numpy()
print("Forward computation", p)
print("trainable variables", model.trainable_variables)
print("all variables", model.variables)

x = tf.ones((2, 2))
lin = Linear(4, 2)
y = lin(x)
print(y)
