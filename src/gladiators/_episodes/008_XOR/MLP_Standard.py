import numpy as np
from src.engine.ActivationFunction import Tanh, Sigmoid, Linear
from src.engine.BaseGladiator import Gladiator
from src.engine.Neuron import Neuron
from src.engine.Utils import smart_format


class MLP_Standard(Gladiator):
    """
    Generalized MLP (Multi-Layer Perceptron) implementation supporting any architecture.
    Uses NumPy for efficient forward and backpropagation.
    """

    def __init__(self, *args, architecture=None):
        super().__init__(*args)
        if architecture is None:
            architecture = []  # Default: Single hidden layer with 2 neurons, 1 output
        self.initialize_neurons(architecture)

        # Assign activation functions (Tanh for hidden layers, Sigmoid for output)
        for layer in Neuron.layers[:-1]:  # Hidden layers
            for neuron in layer:
                neuron.set_activation(Linear)
        for neuron in Neuron.layers[-1]:  # Output layer
            neuron.set_activation(Sigmoid)

    def forward_pass(self, training_sample):
        """
        Executes a forward pass through the entire network.
        Uses NumPy dot product for efficient weight calculations.
        """
        inputs = np.array(training_sample[:-1])  # Extract input features

        for layer in Neuron.layers:
            outputs = []
            for neuron in layer:
                neuron.raw_sum = np.dot(neuron.weights, inputs) + neuron.bias
                neuron.activate()
                outputs.append(neuron.activation_value)
            inputs = np.array(outputs)  # Next layer uses this layer's output

        return Neuron.layers[-1][0].activation_value  # Return final prediction

    def back_pass(self, training_sample, loss_gradient):
        """
        Executes backpropagation using NumPy vectorized operations.
        - Computes error signals layer-by-layer.
        - Adjusts weights and biases using matrix multiplication.
        """
        inputs = np.array(training_sample[:-1])

        # Step 1: Compute error signal for output neurons
        output_layer = Neuron.layers[-1]
        for neuron in output_layer:
            neuron.error_signal = loss_gradient * neuron.activation_gradient

        # Step 2: Backpropagate error signals through hidden layers
        for layer_index in range(len(Neuron.layers) - 2, -1, -1):  # Iterate backwards
            layer = Neuron.layers[layer_index]
            next_layer = Neuron.layers[layer_index + 1]
            for neuron in layer:
                weighted_error = sum(
                    next_neuron.weights[neuron.position] * next_neuron.error_signal
                    for next_neuron in next_layer
                )
                neuron.error_signal = neuron.activation_gradient * weighted_error

        # Step 3: Adjust weights and biases
        prev_activations = inputs  # First hidden layer takes raw inputs
        for layer in Neuron.layers:
            layer_activations = np.array([neuron.activation_value for neuron in layer])
            for neuron in layer:
                neuron.weights += neuron.learning_rate * neuron.error_signal * prev_activations
                neuron.bias += neuron.learning_rate * neuron.error_signal
            prev_activations = layer_activations  # Update for next layer
