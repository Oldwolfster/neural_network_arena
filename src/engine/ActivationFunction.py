import numpy as np


class ActivationFunction:
    """Encapsulates an activation function and its derivative."""
    def __init__(self, function, derivative, name="Custom"):
        self.function = function
        self.derivative = derivative
        self.name = name

    def __call__(self, x):
        """Allows the object to be used as a function."""
        return self.function(x)

    def apply_derivative(self, x):
        """Compute the derivative for backpropagation."""
        return self.derivative(x)
    def __repr__(self):
        return f"ActivationFunction({self.name})"

# Standard activations as objects
Linear = ActivationFunction(
    function=lambda x: x,
    derivative=lambda x: 1,
    name="Linear"
)

Sigmoid = ActivationFunction(
    function=lambda x: 1 / (1 + np.exp(-x)),
    derivative=lambda x: x * (1 - x),  # More efficient if x = sigmoid(x)
    name="Sigmoid"
)

Tanh = ActivationFunction(
    function=np.tanh,
    derivative=lambda x: 1 - np.tanh(x)**2,
    name="Tanh"
)

ReLU = ActivationFunction(
    function=lambda x: np.maximum(0, x),
    derivative=lambda x: np.where(x > 0, 1, 0),
    name="ReLU"
)

LeakyReLU = ActivationFunction(
    function=lambda x, alpha=0.01: np.where(x > 0, x, alpha * x),
    derivative=lambda x, alpha=0.01: np.where(x > 0, 1, alpha),
    name="LeakyReLU"
)
