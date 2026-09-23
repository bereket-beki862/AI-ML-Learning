import numpy as np

arr = np.array([10, 20, 30, 40, 50])

print("Array:", arr)
print("Data type:", arr.dtype)

float_arr = np.array([1.5, 2.5, 3.5])

print("Float array:", float_arr)
print("Data type:", float_arr.dtype)

int_arr = np.array([1, 2, 3], dtype=np.float64)

print("Converted array:", int_arr)
print("Data type:", int_arr.dtype)