from src.engine.Metrics import GladiatorOutput
from src.engine.BaseGladiator import Gladiator
import numpy as np

class Regression_GBS_MultInputs(Gladiator):
    """
    A simple perceptron implementation for accurate regression. (By ChatGPT)
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.bias = .5

    def training_iteration(self, training_data) -> GladiatorOutput:
        inputs          = training_data[:-1]                        # All elements except the last (the inputs)
        target          = training_data[-1]                         # Last element is the target
        prediction      = np.dot(inputs, self.weights) + self.bias  # Calculate prediction as the dot product of inputs and weights plus bias
        error           = target - prediction                       # Calculate error
        self.weights    += self.learning_rate * error * inputs      # Update weights: element-wise adjustment for each input
        self.bias       += self.learning_rate * error               # Update bias
        return prediction                                           # Return Prediction to Superclass

