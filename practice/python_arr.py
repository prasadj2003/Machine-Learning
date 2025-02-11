import numpy as np

# Create a sample 2D array
X = np.array([[1, 2, 3], [4, 1, 3]])

# Print the original array
print("Original Array:")
print(X)

# Example 1: All rows, columns up to 2nd column
print("\nExample 1: All rows, columns up to 2nd column")
print(X[:, :2])

# Example 2: 1st row, all columns
print("\nExample 2: 1st row, all columns")
print(X[1, :])

# Example 3: All rows, columns from 1st column onwards
print("\nExample 3: All rows, columns from 1st column onwards")
print(X[:, 1:])

# Example 4: All rows, center column
print("\nExample 4: All rows, center column")
print(X[:, 1:2])

# Example 5: All rows, all columns with step size of 2
print("\nExample 5: All rows, all columns with step size of 2")
print(X[:, ::2])

# Exercises
print("\nExercises:")
print("0th row, all columns:")
print(X[0, :])
print("All rows, 0th column:")
print(X[:, 0])
print("1st row to 2nd row, all columns:")
print(X[1:2, :])
print("All rows, 1st to 3rd column:")
print(X[:, 1:3])