# -*- coding: utf-8 -*-
import math
import random
import pickle

def Vector(rows):
    cols = 1
    return Matrix(rows, cols)

class Matrix:
    def __init__(self, rows, cols):
        super(Matrix, self).__init__()
        self.rows = rows
        self.cols = cols
        self.data = []

        for i in range(self.rows):
            self.data.append([])
            for j in range(self.cols):
                self.data[i].append(0)

    @staticmethod
    def subtract(a, b):
        # Return a new matrix a-b
        if a.rows != b.rows or a.cols != b.cols:
            print("Columns and Rows must match")
            return

        result = Matrix(a.rows, a.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                result.data[i][j] = (a.data[i][j] - b.data[i][j])
        return result

    def add(self, n):
        if isinstance(n, Matrix):
            if self.rows != n.rows or self.cols != n.cols:
                print("Columns and Rows must match")
                return

            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n.data[i][j]
        else:
            # add n to all matrix elements
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n

    @staticmethod
    def multiplyMatrix(a, b):
        if a.cols != b.rows:
            print("Columns of A must match the rows of B")
            return
        result = Matrix(a.rows, b.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                # Dot product of values in cols
                Sum = 0
                for k in range(a.cols):
                    Sum += a.data[i][k] * b.data[k][j]
                result.data[i][j] = Sum
        return result

    def map(self, func):
        # Apply a function to every element of matrix
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.data[i][j]
                self.data[i][j] = func(val)

    @staticmethod
    def Smap(matrix, func):  # Static version of map function
        result = Matrix(matrix.rows, matrix.cols)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                val = matrix.data[i][j]
                result.data[i][j] = func(val)
        return result

    def multiply(self, n):
        if isinstance(n, Matrix):
            # Hadamard product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n.data[i][j]
        else:
            # Scalar product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n

    @staticmethod
    def fromArray(arr):
        m = Matrix(len(arr), 1)
        for i in range(arr.__len__()):
            m.data[i][0] = arr[i]
        return m

    def toArray(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j])
        return arr

    @staticmethod
    def transpose(matrix):
        # This makes the columns of the new matrix the rows of the original
        result = Matrix(matrix.cols, matrix.rows)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                result.data[j][i] = matrix.data[i][j]
        return result

    def randomize(self, dtype=float):
        # Randomize all the matrix elements
        for i in range(self.rows):
            for j in range(self.cols):
                if dtype == float:
                    # Return a float value 0.1 and 0.9
                    self.data[i][j] = random.random() * 2 - 1
                elif dtype == int:
                    # Return a int value between 0 and 10
                    self.data[i][j] = math.floor(random.random() * 10)

    def log(self):
        print(f"{self.rows} X {self.cols} Matrix:")
        print(self.data)

#        print("{0} X {1} Matrix:".format(self.rows, self.cols))
#        print(" __", end='')
#        
#        for j in range(16*self.cols-1):
#            print(' ', end='')
#        
#        print("__ ")
#        print("|  ", end='')
#
#        for j in range(16*self.cols-1):
#            print(" ", end='')
#        
#        print("  |")
#
#        for i in range(self.rows):
#            print("|  ", end='')
#            for j in range(self.cols):
#                print(f'{self.data[i][j]} ', end='')
#            print(" |")
#        print("|__", end='')
#
#        for j in range(16*self.cols-1):
#            print(" ", end='')
#        print("__|")

    def serialize(self, fname):
        pickle.dump(self, open(fname+'.weights', 'wb'))

    @staticmethod
    def deserialize(data):
        data = pickle.load(open(data+'.weights', 'rb'))
        return data
