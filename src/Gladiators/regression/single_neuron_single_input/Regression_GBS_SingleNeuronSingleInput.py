from src.engine.Metrics import GladiatorOutput
from src.gladiators.BaseGladiator import Gladiator

class Regression_GBS(Gladiator):
    """
    A simple perceptron implementation for accurate regression. (By ChatGPT)
    It is designed for training data that predicts repayment ratio (0.0 to 1.0)
    based on credit score between 0-100, with added noise.
    Includes bias and improved weight adjustment logic for accuracy.
    """

    def __init__(self, *args):
        super().__init__(*args)
        #self.bias = .5
        #self.weights = [0.1]

    def training_iteration(self, training_data) -> GladiatorOutput:
        input = training_data[0]
        target = training_data[-1]
        prediction = (input * self.weights[0]) + self.bias
        error = target - prediction
        self.weights[0] += self.learning_rate * error * input
        self.bias   += self.learning_rate * error  # Bias adjustment


        return prediction

