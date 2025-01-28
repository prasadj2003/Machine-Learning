import numpy as np

def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    """
    Perform gradient descent to optimize w and b.

    Args:
        X (np.ndarray): Input features of shape (m, n), where m is the number of samples and n is the number of features.
        y (np.ndarray): Target values of shape (m,).
        w (np.ndarray): Weights of shape (n,).
        b (float): Bias term.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations to run gradient descent.

    Returns:
        w (np.ndarray): Optimized weights.
        b (float): Optimized bias.
        loss_history (list): History of the loss function values.
    """
    m = X.shape[0]  # Number of samples
    loss_history = []

    for _ in range(num_iterations):
        # Calculate predictions
        y_hat = np.dot(X, w) + b

        # Compute loss (Mean Squared Error)
        loss = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)
        loss_history.append(loss)

        # Compute gradients
        dw = (1 / m) * np.dot(X.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b, loss_history

# Example usage
if __name__ == "__main__":
    # Dummy data
    X = np.array([[1], [2], [3], [4]])  # Input features (m=4 samples, n=1 feature)
    y = np.array([2, 3, 4, 5])          # Target values

    # Initialize weights and bias
    w = np.array([0.0])
    b = 0.0

    # Hyperparameters
    learning_rate = 0.01
    num_iterations = 1000

    # Run gradient descent
    w, b, loss_history = gradient_descent(X, y, w, b, learning_rate, num_iterations)

    print("Optimized weights:", w)
    print("Optimized bias:", b)
    print("Final loss:", loss_history[-1])
