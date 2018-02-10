# -*- coding: utf-8 -*-
from .matrix import Matrix
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_ih.randomize()
        self.weights_ho.randomize()

        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)
        self.bias_h.randomize()
        self.bias_o.randomize()

    def feedforward(self, inputs_array):
        # Generate hidden outputs
        inputs = Matrix.fromArray(inputs_array)
        hidden = Matrix.multiplyMatrix(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        # Activation function
        hidden.map(sigmoid)
        # Generating the output's output!
        output = Matrix.multiplyMatrix(self.weights_ho, hidden)
        output.add(self.bias_o)
        output.map(sigmoid)
        # Sending it back to the caller!
        return output.toArray()

    def setLearningRate(self, learning_rate):
        minimum = 0.1
        maximum = 1.0
        if isinstance(learning_rate, float) and learning_rate > minimum and learning_rate < maximum:
            self.learning_rate = learning_rate
        else:
            raise ValueError("Learning rate must be float and smaller than 1.0 and bigger that 0.1")

    def train(self, input_array, target_array):
        # Generate hidden outputs
        inputs = Matrix.fromArray(input_array)
        hidden = Matrix.multiplyMatrix(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        # Activation function
        hidden.map(sigmoid)
        # Generating the output's output!
        outputs = Matrix.multiplyMatrix(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(sigmoid)

        # Convert array to matrix object
        targets = Matrix.fromArray(target_array)

        # Calculate the error
        # ERROR = TARGETS - OUTPUTS
        output_errors = Matrix.subtract(targets, outputs)

        # Calculate gradient
        gradients = Matrix.Smap(outputs, dsigmoid)
        gradients.multiply(output_errors)
        gradients.multiply(self.learning_rate)

        # Calculate deltas
        hidden_T = Matrix.transpose(hidden)
        weights_ho_deltas = Matrix.multiplyMatrix(gradients, hidden_T)

        # Adjust the weights by deltas
        self.weights_ho.add(weights_ho_deltas)
        # Adjust the bias by its deltas (gradients)
        self.bias_o.add(gradients)

        # Calculate the hidden layer errors
        who_t = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.multiplyMatrix(who_t, outputs)

        # Calculate hidden gradient
        hidden_gradient = Matrix.Smap(hidden, dsigmoid)
        hidden_gradient.multiply(hidden_errors)
        hidden_gradient.multiply(self.learning_rate)

        # Calculate input->hidden deltas
        inputs_T = Matrix.transpose(inputs)
        weight_ih_deltas = Matrix.multiplyMatrix(hidden_gradient, inputs_T)
        self.weights_ih.add(weight_ih_deltas)
        self.bias_h.add(hidden_gradient)
