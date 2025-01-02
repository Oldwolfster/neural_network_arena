from src.engine.BaseGladiator import Gladiator
import numpy as np

class Regression_GBS_MultInputs_MultiNeurons(Gladiator):
    """
    A perceptron implementation for regression supporting any number of neurons.
    Automatically adapts to any number of inputs and neurons, updating weights and biases for all neurons.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.num_neurons = 3  # Default number of neurons; can be adjusted as needed
        self.initialize_neurons(self.num_neurons)  # Dynamically initialize neurons

    def training_iteration(self, training_data) -> float:
        inputs = training_data[:-1]  # All elements except the last (the inputs)
        target = training_data[-1]   # Last element is the target

        # Compute predictions for all neurons
        predictions = [
            np.dot(inputs, neuron.weights) + neuron.bias for neuron in self.neurons
        ]

        # Combine predictions (e.g., sum them up)
        prediction = sum(predictions)

        # Calculate error
        error = target - prediction

        # Update weights and biases for all neurons
        for neuron in self.neurons:
            neuron.weights += self.learning_rate * error * inputs
            neuron.bias += self.learning_rate * error

        return prediction  # Return prediction to superclass
