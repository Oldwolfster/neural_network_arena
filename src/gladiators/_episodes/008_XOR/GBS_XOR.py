from src.engine.BaseGladiator import Gladiator
import numpy as np

class NeuralNetwork_NNA(Gladiator):
    """
    A Neural Network implementation adapted for the Neural Network Arena (NNA).
    Supports multiple layers with a configurable number of neurons.
    """

    def __init__(self, *args):
        super().__init__(*args)
        # Define the architecture: [inputs, hidden1, hidden2, ..., outputs]
        self.layers = [2,1]  # Example architecture with 2 inputs, 3 hidden neurons, and 1 output
        self.alpha = 0.1  # Learning rate
        self.W = []  # List to hold weight matrices

        # Initialize weight matrices
        for i in np.arange(0, len(self.layers) - 2):
            w = np.random.randn(self.layers[i] + 1, self.layers[i + 1] + 1) / np.sqrt(self.layers[i])
            self.W.append(w)

        w = np.random.randn(self.layers[-2] + 1, self.layers[-1]) / np.sqrt(self.layers[-2])
        self.W.append(w)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def training_iteration(self, training_data):
        inputs = training_data[:-1]  # Inputs from the training data
        target = training_data[-1]  # Target value from the training data

        # Add bias to the inputs
        inputs = np.append(inputs, 1)  # Append a bias input of 1

        # Forward pass
        A = [np.atleast_2d(inputs)]  # Activations list
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)

        # Compute error
        error = A[-1] - target

        # Backpropagation
        D = [error * self.sigmoid_deriv(A[-1])]
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        D = D[::-1]  # Reverse deltas

        # Update weights
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

        return float(A[-1])  # Return prediction

    def predict(self, inputs):
        inputs = np.append(inputs, 1)  # Append a bias input of 1
        for layer in np.arange(0, len(self.W)):
            inputs = self.sigmoid(np.dot(inputs, self.W[layer]))
        return float(inputs)
