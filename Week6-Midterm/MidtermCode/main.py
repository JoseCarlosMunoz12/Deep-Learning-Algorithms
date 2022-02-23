import math

import numpy as np


class Layer:
    def __init__(self, weights, function_type):
        self.weights = weights
        self.a_function = function_type

    def sigmoid_f(self, z):
        sig = lambda t: 1 / (1 + math.exp(-t))
        return np.array([sig(xi) for xi in z])

    def multiply(self, a_input):
        weights_T = np.matrix.transpose(self.weights)
        print(weights_T)
        z = np.matmul(weights_T, a_input)
        print(z)
        return self.sigmoid_f(z)


def main(name):
    with open(name) as f:
        lines = f.readlines()
    layers = []
    cur_layer = np.array(())
    num_rows = 0
    num_cols = 0
    for line in lines:
        if line[0] == '[':
            cur_layer = np.empty((int(line[1]), int(line[3])))
            num_rows = int(line[1])
            num_cols = int(line[3])
        elif line[0] == 'f':
            layers.append(Layer(cur_layer, line))
        else:
            count = 0
            nums = line.split(',')
            for j in range(num_cols):
                for i in range(num_rows):
                    cur_layer[i][j] = float(nums[count])
                    count += 1
    point = np.array([[0.82],
                      [0.23]])
    for layer in layers:
        point = layer.multiply(point)
        print(point)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('FF.txt')

