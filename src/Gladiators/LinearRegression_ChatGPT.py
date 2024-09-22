from src.Arena import *
from src.Metrics import Metrics, IterationData
from src.Gladiator import Gladiator

class LinearRegression_ChatGPT(Gladiator):
    """
    A simple linear regression implementation for educational purposes.
    This class serves as a template for more complex implementations.
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, *args):
        super().__init__(number_of_epochs, metrics, *args)
        self.bias = 0.0  # Initialize bias term

    def training_iteration(self, training_data) -> IterationData:
        credit_score = training_data[0]
        result = training_data[1]
        prediction = self.predict(credit_score)
        loss = self.compare(prediction, result)
        adjustment_weight, adjustment_bias = self.adjust_parameters(credit_score, result)
        new_weight = self.weight + adjustment_weight
        new_bias = self.bias + adjustment_bias
        data = IterationData(
            prediction=prediction,
            adjustment=adjustment_weight,
            weight=self.weight,
            new_weight=new_weight,
            bias=self.bias,
            new_bias=new_bias
        )
        self.weight = new_weight
        self.bias = new_bias
        return data  # Return loss to accumulate for epoch summary

    def predict(self, credit_score: float) -> float:
        return credit_score * self.weight + self.bias  # Linear prediction

    def compare(self, prediction: float, result: float) -> float:
        return (result - prediction)  # Return just the difference, not squared

    def adjust_parameters(self, credit_score: float, result: float) -> tuple:
        prediction = self.predict(credit_score)
        error = prediction - result
        adjustment_weight = -2 * error * credit_score * self.learning_rate  # Gradient descent for weight
        adjustment_bias = -2 * error * self.learning_rate  # Gradient descent for bias
        return adjustment_weight, adjustment_bias
