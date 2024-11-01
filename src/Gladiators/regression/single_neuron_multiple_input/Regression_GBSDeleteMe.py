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
        inputs = training_data[:-1]    # All elements except the last (the inputs)
        target = training_data[-1]     # Last element is the target

        # Calculate prediction as the dot product of inputs and weights plus bias
        prediction = np.dot(inputs, self.weights) + self.bias

        # Calculate error
        error = target - prediction

        # Update weights: element-wise adjustment for each input
        new_weights = self.weights + self.learning_rate * error * inputs

        # Update bias
        new_bias = self.bias + self.learning_rate * error

        # Output object containing results of this iteration
        gladiator_output = GladiatorOutput(
            prediction=prediction
        )

        # Update model parameters with new weights and bias
        self.weights = new_weights
        self.bias = new_bias

        return gladiator_output

    def predict(self, credit_score: float) -> float:
        """
        Predict repayment ratio based on the credit score and current weights.
        Includes bias term.
        """
        return (credit_score * self.weight) + self.bias  # Adding bias term to the prediction

    def compare(self, prediction: float, target: float) -> float:
        """
        Calculate the error: difference between target and prediction.
        """
        return target - prediction

    def adjust_weight(self, error: float, credit_score: float) -> float:
        """
        Adjust weight based on the error, learning rate, and input.
        Following gradient descent principles: adjustment = error * learning_rate * input
        """
        return self.learning_rate * error * credit_score
