# -*- coding: utf-8 -*-
import math
import random


class Matrix:
    def __init__(self, rows, cols):
        # The user will input the number
        # of rows and cols to create a matrix
        # and store it to a variable
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
        result = Matrix(a.rows, a.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                result.data[i][j] = (a.data[i][j] - b.data[i][j])
        return result

    def add(self, n):
        if isinstance(n, Matrix):
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
            return None
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
        print(self.data)
