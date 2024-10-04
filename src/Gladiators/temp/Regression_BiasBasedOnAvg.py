from src.Engine import *
from src.Metrics import GladiatorOutput
from src.BaseGladiator import Gladiator


class Regression_BiasBasedOnAvg(Gladiator):
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

        # Step 1: Guess (prediction)
        prediction = input * self.weight + self.bias

        # Step 2: Check guess (error)
        error = result - prediction

        # Step 3: Adjust weight based on the specific sample
        weight_adjustment = error * self.learning_rate
        new_weight = self.weight + weight_adjustment
        # Calculate the current average error (running average) based on total error so far
        if self.metrics_mgr.iteration_num > 0:  # Ensure there's data to calculate average error
            average_error = self.metrics_mgr.summary.total_error / self.metrics_mgr.iteration_num
        else:
            average_error = 0  # No change if there are no iterations yet

        # Adjust bias on each iteration based on the running mean error
        bias_adjustment = self.learning_rate * average_error
        new_bias = self.bias + bias_adjustment

        # Prepare GladiatorOutput to log adjustments
        gladiator_output = GladiatorOutput(
            prediction=prediction,
            adjustment=weight_adjustment,
            weight=self.weight,
            new_weight=new_weight,
            bias=self.bias,
            new_bias=new_bias
        )

        # Update weight and bias for the next iteration
        self.weight = new_weight
        self.bias = new_bias

        return gladiator_output
