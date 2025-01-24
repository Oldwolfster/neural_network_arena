import numpy as np
import matplotlib.pyplot as plt

# Define a simple dataset
X = np.array([[2, 3], [1, 1], [4, 5], [6, 5], [3, 4], [7, 8]])
y = np.array([0, 0, 0, 1, 0, 1])  # Labels

# Initialize weights and bias
weights = np.random.randn(2)
bias = np.random.randn()

# Perceptron forward function
def perceptron_forward(x, weights, bias):
    return np.dot(weights, x) + bias

# Training loop (simple perceptron training for illustration)
learning_rate = 0.01
for epoch in range(100):
    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]
        z = perceptron_forward(x_i, weights, bias)
        a = 1 if z >= 0 else 0
        error = y_i - a
        weights += learning_rate * error * x_i
        bias += learning_rate * error

# Plotting the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o')

# Plotting the decision boundary
x_values = np.linspace(0, 8, 100)
y_values = -(weights[0] * x_values + bias) / weights[1]
plt.plot(x_values, y_values, label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Decision Boundary of Simple Perceptron')
plt.show()



