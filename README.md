# Simple Neural Network

neural network in python without using any external library

all is implemented manually

i am just creating my first neural network library


# Docs

#### classes
`NeuralNetwork` takes 3 required params and one optional-  NeuralNetwork(int(input_nodes), int(hidden_nodes), int(output_nodes), learning_rate=.1)

`Matrix` takes 2 required params - Matrix(int(rows), int(cols))

#### methods

###### Matrix
`subtract`: takes to param subtract(a, b) - subtract 2 Matrix object

`add`: takes 1 param add(n) - if `n` is instance of matrix it returns Hadamard Product else returns scalar product

`multiplyMatrix`: takes 2 params multiplyMatrix() - multiply 2 Matrix object

`map`: takes 1 param - Apply a function to every element of Matrix object

`Smap`: takes 2 params - Smap(a, b) static version of map method

`multiply`: takes 1 param add(n) - if `n` is instance of matrix it returns Hadamard Product else returns scalar product

`fromArray`: takes 1 param - convert array to Matrix object

`toArray`: takes 1 param - convert object Matrix to array

`transpose`: takes 1 param - This makes the columns of the new matrix the rows of the original

`randomize`: randomize matrix object

`log`: print the data of the givin matrix

###### NeuralNetwork
`feedforward`: takes 1 param - feedforward(input_array) - returns output in array

`train`: takes 2 param - train(input_array, target_array) - train the Neural Network

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
# [[-0.26396268483049146, 0.3837936231559904, -0.9863464021672874], [-0.6479179674474989, 0.26713230080347317, 0.061410519618629644]] of course your answer will vary
m1.randomize(dtype=int)
print(m1.data)
# [[6, 5, 9], [2, 4, 7]] of course your answer will vary
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
output = nn.feedforward(inputs)
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
outputs = nn.feedforward(inputs)
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
        print(training_data[data].get("inputs"), training_data[data].get('targets'))

print(nn.feedforward([1, 0]))
print(nn.feedforward([0, 1]))
print(nn.feedforward([0, 0]))
print(nn.feedforward([1, 1]))
```
Now you can see the result

#### LICENSE
GPL3
