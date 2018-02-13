# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
import random

sys.path.append('../NeuralNetwork')

from mnist import loadMNIST
from nn import NeuralNetwork
from nn import sigmoid


def image_show(image):
    plt.imshow(image, interpolation='nearest', cmap=plt.cm.gray_r)
    plt.show()

def ascii_show(image):
    for y in image:
         row = ""
         for x in y:
             row += '{0: <4}'.format(x)
         print(row)

def getOutput(label):
    targets = np.zeros(10, dtype=int)
    targets[label] = 1
    print("output nodes: {0}".format(targets))

# Loading mnist training data and testing data
training_data = list(loadMNIST('training', lambda: print("training dataset loaded without errors")))
testing_data = list(loadMNIST('testing', lambda: print("training dataset loaded without errors\n")))

# Number of pixels | Number of Hidden Nodes | Number of outputs | learning_rate
nn = NeuralNetwork(784, 64, 10, learning_rate=0.1, activation_function=sigmoid)

label, pixels = training_data[random.randint(0, len(training_data))]

# Print Training Data Length
print("Training_data length: {0}\n".format(len(training_data)))
# Print Pixles Shape
print("Pixles Shape: {0}\n".format(pixels.shape))
# Print the output
getOutput(label)
# Show image in Terminal
ascii_show(pixels)
# Show image in matplotlib
image_show(pixels)

# TODO: train the neural network
# TODO: test the neural network
# DEBUG: make the code more cleaner then debug it
