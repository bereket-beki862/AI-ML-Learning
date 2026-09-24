import numpy as np

matrix1 = np.array([
    [1, 2],
    [3, 4]
])

matrix2 = np.array([
    [5, 6],
    [7, 8]
])

print("Matrix 1:")
print(matrix1)

print("Matrix 2:")
print(matrix2)

print("Addition:")
print(matrix1 + matrix2)

print("Multiplication:")
print(matrix1 * matrix2)

print("Matrix multiplication:")
print(np.dot(matrix1, matrix2))