import numpy as np

# Define a simple linear model: y = w * x
def predict(x, w):
    return w * x

# Mean Squared Error Loss
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient of the loss function with respect to w
def compute_gradient(x, y_true, y_pred):
    return -2 * np.mean(x * (y_true - y_pred))

# Simple weight update function
def update_weights(w, gradient, learning_rate):
    return w - learning_rate * gradient

# Example data
x = np.array([1, 2, 3, 4])  # Features
y_true = np.array([2, 4, 6, 8])  # True labels (target)

# Initialize weight
w = 0.0  # Start with some initial guess for the weight
learning_rate = 0.01  # Small learning rate

# Training loop for a few iterations
for i in range(100):
    y_pred = predict(x, w)  # Predict output using the current weight
    loss = compute_loss(y_true, y_pred)  # Calculate the loss
    gradient = compute_gradient(x, y_true, y_pred)  # Calculate the gradient
    w = update_weights(w, gradient, learning_rate)  # Update the weight

    # Print the loss and the updated weight
    if i % 10 == 0:
        print(f"Iteration {i}: Loss = {loss:.4f}, Weight = {w:.4f}")