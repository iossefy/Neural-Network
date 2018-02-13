# Simple Neural Network

neural network in python without using any external library

all is implemented manually

i am just creating my first neural network library


# Documentation

* `NeuralNetwork` - The neural network class
  * `predict(input_array)` - Returns the output of a neural network
  * `train(input_array, target_array)` - Trains a neural network
  * `setLearningRate(learning_rate)` - setting the learning rate
  * `setActivationFunction(func)` - setting the activation function

* `Matrix` - The matrix class
  * `subtract(a, b)` - return a matrix result from subtracting 2 matrix objects
  * `add(n)` - add return sum of 2 matrix objects
  * `multiplyMatrix(a, b)` - return a matrix result from multiplying to matrix objects
  * `map(func)` - Apply a function to every element of matrix
  * `Smap(matrix, func)` - static version of map
  * `multiply(n)` - result a Hadamard product or scalar product
  * `fromArray(arr)` - convert array to matrix
  * `toArray` - convert matrix to array
  * `transpose(matrix)` - This makes the columns of the new matrix the rows of the original
  * `randomize(dtype=float)` - randomize all matrix elements
  * `log` - print matrix data

# Code Examples

###### Matrix
matrix is a numpy like library
```python
from NeuralNetwork.matrix import Matrix
# create instance of matrix with 2 rows and 3 columns
m1 = Matrix(2, 3)

# Show Data
print(m1.data)
# [[0, 0, 0], [0, 0, 0]]

# Randomize values of m1 (Matrix)
m1.randomize(dtype=float)
print(m1.data)
# [[-0.26396268483049146, 0.3837936231559904, -0.9863464021672874], [-0.6479179674474989, 0.26713230080347317, 0.061410519618629644]]
m1.randomize(dtype=int)
print(m1.data)
# [[6, 5, 9], [2, 4, 7]]
# Change rows and cols manually
m1.data[0][1] = 0
print(m1.data)
# [[6, 0, 9], [2, 4, 7]]
```

Matrix Methods

```python
from NeuralNetwork.matrix import Matrix
# create 2 matrix object
m1 = Matrix(2, 3)
m2 = Matrix(3, 2)
# Randomize values
m1.randomize(dtype=int)
m2.randomize(dtype=int)
print(m1.data)
# [[8, 3, 9], [7, 6, 6]]
print(m2.data)
# [[2, 8], [6, 2], [1, 0]]
# Transpose
m3 = Matrix.transpose(m2)
print(m3.data)
# [[2, 6, 1], [8, 2, 0]]
# Matrix multiplication
m4 = Matrix.multiplyMatrix(m1, m2)
print(m4.data)
# [[43, 70], [56, 68]]
```

###### Neural Network
```python
from NeuralNetwork.nn import NeuralNetwork
nn = NeuralNetwork(2, 2, 1, learning_rate=0.1)
inputs = [1, 0]
output = nn.predict(inputs)
print(output)
```

you can train it and feedforward it

```python
from NeuralNetwork.nn import NeuralNetwork
# Input nodes, Hidden nodes, Output nodes, learning_rate
nn = NeuralNetwork(2, 2, 1, learning_rate=0.1)

inputs = [1, 0]
targets = [1]

# Train the neural network
nn.train(inputs, targets)
outputs = nn.predict(inputs)
print(outputs)
# [0.30405332078202085] # it will show you something like that
```

you can load the data from json file

```python

from NeuralNetwork.nn import NeuralNetwork
nn = NeuralNetwork(2, 2, 1, learning_rate=0.1)

training_data = [
    {
        'inputs': [0, 1],
        'targets': [1],
    },
    {
        'inputs': [1, 0],
        'targets': [1],
    },
    {
        'inputs': [0, 0],
        'targets': [0],
    },
    {
        'inputs': [1, 1],
        'targets': [0],
    },
]

for i in range(1000):
    for data in range(len(training_data)):
        nn.train(training_data[data].get('inputs'), training_data[data].get('targets'))

print(nn.predict([1, 0]))
print(nn.predict([0, 1]))
print(nn.predict([0, 0]))
print(nn.predict([1, 1]))
```
Now you can see the result

#### LICENSE
GPL3
