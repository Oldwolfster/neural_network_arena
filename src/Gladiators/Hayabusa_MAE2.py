from src.ArenaEngine import *
from src.Metrics import GladiatorOutput
from src.BaseGladiator import Gladiator


class Hayabusa_MAE2(Gladiator):
    """
    A simple perceptron using Regression
    This class serves as a template for more complex implementations.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)
        self.bias = .5

    def training_iteration(self, training_data) -> GladiatorOutput:
        input = training_data[0]
        result = training_data[1]
        prediction  = input * self.weight + self.bias                               # Step 1) Guess
        error       = result - prediction                  # Step 2) Check guess, if wrong, how much?
        adjustment  = error * self.learning_rate                          # Step 3) Adjust(Calc)
        new_weight  = self.weight + adjustment                          # Step 3) Adjust(Apply)
        new_bias    = self.bias + adjustment
        gladiator_output = GladiatorOutput(
            prediction=prediction,
            adjustment=adjustment,
            weight=self.weight,
            new_weight=new_weight,
            bias=self.bias,
            new_bias=new_bias
        )
        self.weight = new_weight
        self.bias = new_bias
        return gladiator_output

"""
    def predict(self, input: float) -> int:
        return input * self.weight + self.bias                               # Rounded to 7 decimals to avoid FP errors

    def compare(self, prediction: float, result: float) -> float:            # Calculate the Loss
        return result - prediction

    def adjust_weight(self, loss: float) -> float:                       # Apply learning rate to loss.
        return loss * self.learning_rate
"""