
from src.engine.BaseGladiator import Gladiator
import numpy as np

class Regression_GBS_MultInputs(Gladiator):
    """
    A simple perceptron implementation for accurate regression. (By ChatGPT)
    Automatically adapts to any number of inputs and processes with the fancy parallel techinques
    """

    def __init__(self, *args):
        super().__init__(*args)
        #self.bias = .5

    def training_iteration(self, training_data) -> float:

        inputs          = training_data[:-1]                        # All elements except the last (the inputs)
        target          = training_data[-1]                         # Last element is the target
        prediction      = np.dot(inputs, self.weights) + self.bias  # Calculate prediction as the dot product of inputs and weights plus bias
        error           = target - prediction                       # Calculate error
        print(f"PREDICTION in GBS_ONE_NEURON{prediction}\tself.bias={self.bias}")
        """
        print(f"weights dtype: {self.weights.dtype}")
        print(f"learning_rate type: {type(self.learning_rate)}, value: {self.learning_rate}")
        print(f"error type: {type(error)}, value: {error}")
        print(f"inputs dtype: {inputs.dtype}")
        """


        self.weights    += self.learning_rate * error * inputs      # Update weights: element-wise adjustment for each input
        self.bias       += self.learning_rate * error               # Update bias
        return prediction                                           # Return Prediction to Superclass

