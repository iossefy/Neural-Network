# -*- coding: utf-8 -*-
from matrix import Matrix
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

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
        outputs = self.feedforward(inputs)
        # Covert array to matrix object
        outputs = Matrix.fromArray(outputs)
        targets = Matrix.fromArray(targets)
        # ERROR = TARGETS - OUTPUTS
        output_errors = Matrix.subtract(targets, outputs)

        # Calculate hidden layer errors
        who_t = Matrix.transpose(self.weights_ho)
        hidden_erros = Matrix.multiplyMatrix(who_t, output_errors)

        # print("Hidden errors: {0}".format(str(hidden_erros.data)))
        # print("Output errors: {0}".format(str(output_errors.data)))
        # print("Targets: {0}".format(str(targets.data)))
        # print("Outputs: {0}".format(str(outputs.data)))
