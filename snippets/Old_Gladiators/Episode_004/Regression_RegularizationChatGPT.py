from src.ArenaEngine import *
from src.Metrics import GladiatorOutput
from src.BaseGladiator import Gladiator

class Regression_RegularizationChatGPT(Gladiator):
    """
    A simple perceptron implementation for accurate regression.
    It is designed for training data that predicts repayment ratio (0.0 to 1.0)
    based on credit score between 0-100, with added noise.
    Includes bias, weight clipping, and regularization for stable training.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)
        self.bias = 0.5  # Initializing bias with a reasonable starting value

    def training_iteration(self, training_data) -> GladiatorOutput:
        credit_score = training_data[0]
        repay_ratio = training_data[1]

        # Step 1: Prediction
        prediction = self.predict(credit_score)

        # Step 2: Calculate error
        error = self.compare(prediction, repay_ratio)

        # Step 3: Adjust weight with clipping
        adjustment = self.adjust_weight(error, credit_score)
        new_weight = self.weight + adjustment
        new_weight = max(min(new_weight, 1.0), -1.0)  # Clipping weights to prevent explosion

        # Adjust the bias
        bias_adjustment = self.learning_rate * error  # Bias adjustment based on the error
        new_bias = self.bias + bias_adjustment

        # Output object containing results of this iteration
        gladiator_output = GladiatorOutput(
            prediction=prediction,
            adjustment=adjustment,
            weight=self.weight,
            new_weight=new_weight,
            bias=self.bias,
            new_bias=new_bias
        )

        # Update model parameters
        self.weight = new_weight
        self.bias = new_bias  # Update the bias

        return gladiator_output

    def predict(self, credit_score: float) -> float:
        """
        Predict repayment ratio based on the credit score and current weights.
        Includes bias term in the prediction.
        """
        return (credit_score * self.weight) + self.bias  # Bias term included in prediction

    def compare(self, prediction: float, target: float) -> float:
        """
        Calculate the error: difference between target and prediction.
        """
        return target - prediction

    def adjust_weight(self, error: float, credit_score: float) -> float:
        """
        Adjust weight based on the error, learning rate, and input.
        Regularization is included to stabilize weight growth.
        """
        regularization_term = 0.01 * self.weight  # L2 regularization to penalize large weights
        return (self.learning_rate * error * credit_score) - regularization_term
