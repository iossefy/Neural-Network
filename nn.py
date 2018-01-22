# -*- coding: utf-8 -*-
from matrix import Matrix
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dsigmoid(x):
    y = sigmoid(x)
    return y * (1 - y)


def alreadySigmoided(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate

        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_ih.randomize()
        self.weights_ho.randomize()

        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)
        self.bias_h.randomize()
        self.bias_o.randomize()

    def feedforward(self, inputArray):
        # Generate hidden outputs
        inputs = Matrix.fromArray(inputArray)
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

    def train(self, inputs, targets):
        # Generate hidden outputs
        outputs = self.feedforward(inputs)

        inputs = Matrix.fromArray(inputs)
        hidden = Matrix.multiplyMatrix(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        # Activation function
        hidden.map(sigmoid)
        # Generating the output's output!
        outputs = Matrix.multiplyMatrix(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(sigmoid)
        # outputs = self.feedforward(inputs)
        # Covert array to matrix object
        targets = Matrix.fromArray(targets)

        # ERROR = TARGETS - OUTPUTS
        output_errors = Matrix.subtract(targets, outputs)
        # Change hidden to output weights gredient descent
        d_outputs = Matrix.Smap(outputs, alreadySigmoided)
        # multiply things with other things
        d_outputs.multiply(output_errors)
        d_outputs.multiply(self.lr)
        hidden_T = Matrix.transpose(hidden)
        delta_weights = Matrix.multiplyMatrix(d_outputs, hidden_T)
        Matrix.log(delta_weights)
        self.weights_ho.add(delta_weights)
        # Calculate hidden layer errors
        who_t = Matrix.transpose(self.weights_ho)
        hidden_erros = Matrix.multiplyMatrix(who_t, output_errors)

        # print("Hidden errors: {0}".format(str(hidden_erros.data)))
        # print("Output errors: {0}".format(str(output_errors.data)))
        # print("Targets: {0}".format(str(targets.data)))
        # print("Outputs: {0}".format(str(outputs.data)))
