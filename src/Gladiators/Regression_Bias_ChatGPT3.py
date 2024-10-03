from src.ArenaEngine import *
from src.Metrics import GladiatorOutput
from src.BaseGladiator import Gladiator

class Regression_Bias_ChatGPT3(Gladiator):
    """
    A simple perceptron implementation for accurate regression.
    It is designed for training data that predicts repayment ratio (0.0 to 1.0)
    based on credit score between 0-100, with added noise.
    Includes bias and improved weight adjustment logic for accuracy.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)
        self.bias = .5

    def training_iteration(self, training_data) -> GladiatorOutput:
        input = training_data[0]

        result = training_data[1]
        prediction = self.predict(input)            # Step 1: Prediction
        error = self.compare(prediction, result)  # Step 2: Calculate error (MSE can be applied here later, starting with simple error)


        adjustment = self.adjust_weight(error, input)  # Step 3: Calculate weight and bias adjustment
        new_weight = self.weight + adjustment
        new_bias = self.bias + (self.learning_rate * error)  # Bias adjustment

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
