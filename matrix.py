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

    def multiply(self, n):
        # Scalar product
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] *= n

    def transpose(self):
        # This makes the columns of the new matrix the rows of the original
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def randomize(self):
        # Randomize all the matrix elements
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = math.floor(random.random() * 10)
