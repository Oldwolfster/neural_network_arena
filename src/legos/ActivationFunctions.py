import numpy as np

class StrategyActivationFunction:
    """Encapsulates an activation function and its derivative."""
    def __init__(self, function, derivative, name="Custom"):
        self.function = function
        self.derivative = derivative
        self.name = name

    def __call__(self, x):
        """Allows the object to be used as a function."""
        return self.function(x)

    def __repr__(self):
        """Custom representation for debugging."""
        return self.name

    def apply_derivative(self, x):
        """Compute the derivative for backpropagation."""
        return self.derivative(x)

# Standard activations as objects
Activation_NoDamnFunction = StrategyActivationFunction(
    function=lambda x: x,
    derivative=lambda x: 1,
    name="None"
)

Activation_Sigmoid = StrategyActivationFunction(
    function=lambda x: 1 / (1 + np.exp(-x)),
    derivative=lambda x: x * (1 - x),  # More efficient if x = sigmoid(x)
    name="Sigmoid"
)

Activation_Tanh = StrategyActivationFunction(
    function=np.tanh,
    derivative=lambda x: 1 - np.tanh(x)**2,
    name="Tanh"
)

Activation_ReLU = StrategyActivationFunction(
    function=lambda x: np.maximum(0, x),
    derivative=lambda x: np.where(x > 0, 1, 0),
    name="ReLU"
)

Activation_LeakyReLU = StrategyActivationFunction(
    function=lambda x, alpha=0.01: np.where(x > 0, x, alpha * x),
    derivative=lambda x, alpha=0.01: np.where(x > 0, 1, alpha),
    name="LeakyReLU"
)

def get_activation_derivative_formula(activation_name: str) -> str:
    """
    Returns the function and derivative formulas as strings for a given activation function.

    Parameters:
        activation_name (str): The name of the activation function.

    Returns:
        dict: A dictionary containing the formulas for the function and its derivative.
    """
    formulas = {
        "Sigmoid":  "σ'(x) = σ(x) * (1 - σ(x))",
        "Tanh":"tanh'(x) = 1 - tanh(x)^2",
        "ReLU":"ReLU'(x) = 1 if x > 0 else 0",
        "LeakyReLU": "LeakyReLU'(x) = 1 if x > 0 else α"
   }
    return formulas.get(activation_name)


def get_activation_formula(activation_name: str) -> dict:
    """
    Returns the function and derivative formulas as strings for a given activation function.

    Parameters:
        activation_name (str): The name of the activation function.

    Returns:
        dict: A dictionary containing the formulas for the function and its derivative.
    """
    formulas = {
        "Sigmoid": {
            "function": "σ(x) = 1 / (1 + e^(-x))",
            "derivative": "σ'(x) = σ(x) * (1 - σ(x))"
        },
        "Tanh": {
            "function": "tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))",
            "derivative": "tanh'(x) = 1 - tanh(x)^2"
        },
        "ReLU": {
            "function": "ReLU(x) = max(0, x)",
            "derivative": "ReLU'(x) = 1 if x > 0 else 0"
        },
        "LeakyReLU": {
            "function": "LeakyReLU(x) = x if x > 0 else α * x",
            "derivative": "LeakyReLU'(x) = 1 if x > 0 else α"
        }
    }

    return formulas.get(activation_name, {"function": "Unknown", "derivative": "Unknown"})
