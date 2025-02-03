import numpy as np

from src.engine.ActivationFunction import *


class Neuron:
    """
    Represents a single neuron with weights, bias, and an activation function.
    """
    layers = []  # Shared across all Gladiators, needs resetting per run
    def __init__(self, nid: int, num_of_weights: int, learning_rate: float, layer_id: int = 0, activation = None):
        #print(f"creating neuron - nid={nid}")
        self.nid = nid
        self.layer_id = layer_id  # Add layer_id to identify which layer the neuron belongs to
        self.num_of_weights = num_of_weights
        self.weights_before = np.array([(nid + 1) * 10 for _ in range(num_of_weights)], dtype=np.float64)
        self.weights = self.weights_before.copy()       #Done so it's available to create view
        self.neuron_inputs = np.zeros_like(self.weights)
        self.bias = float(nid +1)                      # Explicitly set as float
        self.bias_before = self.bias                    #Done so it's available to create view
        self.learning_rate = learning_rate
        self.raw_sum = 0.0
        self.activation_value = 0.0
        self.activation =  activation or Linear         # function
        self.activation_gradient = 0.0  # Store activation gradient from forward pass
        self.error_signal = 6969.69
        self.weight_adjustments = ""
        self.error_signal_calcs = ""


        # ✅ Ensure activation is never None
        self.activation = activation if activation is not None else Linear
        self.activation_name = self.activation.name  # ✅ No more AttributeError

        # Ensure layers list is large enough to accommodate this layer_id
        while len(Neuron.layers) <= layer_id:
            Neuron.layers.append([])

        # Add neuron to the appropriate layer and set its position
        Neuron.layers[layer_id].append(self)
        self.position = len(Neuron.layers[layer_id]) - 1  # Zero-based position within layer


    def activate(self):
        """Applies the activation function."""
        self.activation_value = self.activation(self.raw_sum)
        self.activation_gradient = self.activation.apply_derivative(self.activation_value)  # Store gradient!

    def set_activation(self, activation_function):
        """Dynamically update the activation function."""
        self.activation = activation_function
        self.activation_name = activation_function.name

    def compute_gradient(self):
        """Use the derivative for backpropagation."""
        return self.activation.apply_derivative(self.activation_value)
    @classmethod
    def reset_layers(cls):
        """ Clears layers before starting a new Gladiator. """
        cls.layers = []

    """
    the below methods restrict experimenting to much.
    def forward(self, inputs: np.ndarray) -> float:
        "" "
        Compute the neuron's output given the inputs.
        "" "
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(weighted_sum)

    def update(self, inputs: np.ndarray, error: float):
        "" "
        Update the weights and bias based on the error.
        "" "
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error
"""

import numpy as np





# 1. Uniform Random Initialization (between -1 and 1)
def initialize_uniform_random(neurons):
    for neuron in neurons:
        neuron.weights = np.random.uniform(-1, 1, size=len(neuron.weights))
        neuron.bias = np.random.uniform(-1, 1)

# 2. Normal Distribution Initialization (mean=0, std=1)
def initialize_normal_random(neurons):
    for neuron in neurons:
        neuron.weights = np.random.normal(0, 1, size=len(neuron.weights))
        neuron.bias = np.random.normal(0, 1)

# 3. Xavier/Glorot Initialization (good for sigmoid/tanh activations)
def initialize_xavier(neurons):
    for neuron in neurons:
        limit = np.sqrt(6 / (len(neuron.weights) + 1))  # +1 for bias
        neuron.weights = np.random.uniform(-limit, limit, size=len(neuron.weights))
        neuron.bias = np.random.uniform(-limit, limit)

# 4. He Initialization (good for ReLU activations)
def initialize_he(neurons):
    for neuron in neurons:
        limit = np.sqrt(2 / len(neuron.weights))
        neuron.weights = np.random.normal(0, limit, size=len(neuron.weights))
        neuron.bias = np.random.normal(0, limit)

# 5. Small Random Values (close to zero)
def initialize_small_random(neurons):
    for neuron in neurons:
        neuron.weights = np.random.randn(len(neuron.weights)) * 0.01
        neuron.bias = np.random.randn() * 0.01
