from src.engine.Metrics import GladiatorOutput
from src.engine.BaseGladiator import Gladiator

class Regression_MSE(Gladiator):
    """
    A simple perceptron implementation for accurate regression.
    It includes bias and supports multiple loss functions for accurate weight adjustment.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr,  *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)
        self.bias = .5
        self.loss_function = "MSE"

    def calculate_loss_gradient(self, error: float, input: float) -> float:
        """
        Compute the gradient based on the selected loss function (MSE, MAE).
        """
        if self.loss_function == 'MSE':
            # Mean Squared Error: Gradient is error * input
            return error * input
        elif self.loss_function == 'MAE':
            # Mean Absolute Error: Gradient is sign of the error * input
            return (1 if error >= 0 else -1) * input
        else:
            # Default to MSE if no valid loss function is provided
            return error * input

    def training_iteration(self, training_data) -> GladiatorOutput:
        input = training_data[0]
        target = training_data[1]
        prediction = (input * self.weight) + self.bias
        error = target - prediction

        # Calculate the gradient based on the chosen loss function
        gradient = self.calculate_loss_gradient(error, input)

        # Weight and bias updates
        new_weight = self.weight + self.learning_rate * gradient
        new_bias = self.bias + self.learning_rate * error  # Bias adjustment remains simple

        # Output object containing results of this iteration
        gladiator_output = GladiatorOutput(
            prediction=prediction,
            adjustment=new_weight - self.weight,
            weight=self.weight,
            new_weight=new_weight,
            bias=self.bias,
            new_bias=new_bias
        )

        # Update model parameters
        self.weight = new_weight
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

