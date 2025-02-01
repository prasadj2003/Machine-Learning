import numpy as np

# with np.dot

# Given dataset
X = np.array([[-2.509], [9.014], [4.640], [1.973], [-6.880]])
y = np.array([-0.353, 33.445, 21.103, 8.944, -14.078])

# Initialize w and b
w, b = 0.0, 0.0

# Define learning rate
alpha = 0.01

# Number of samples
m = X.shape[0]

print(f"Number of samples (m): {m}")

# Gradient Descent Function
def gradient_descent(X, y, w, b, m, alpha, epochs=10):
    for i in range(epochs):
        y_pred = w * X.flatten() + b  # Predictions
        print(f"type of y: {type(y_pred)}")
        print(y_pred)

        # Compute gradients using np.dot
        dj_dw = (1/m) * np.dot(X.flatten(), (y_pred - y))  # Equivalent to summing element-wise
        dj_db = (1/m) * np.sum(y_pred - y)  # Scalar sum

        # Update weights
        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b

# Run gradient descent
w_optimized, b_optimized = gradient_descent(X, y, w, b, m, alpha)

# Display results
print(f"Optimized Weight: {w_optimized}")
print(f"Optimized Bias: {b_optimized}")
