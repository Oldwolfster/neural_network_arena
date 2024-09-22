from src.Arena import *
from src.Metrics import Metrics, IterationData
from src.Gladiator import Gladiator

class LinearRegression_Claude(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
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
        adjustment_weight, adjustment_bias = self.adjust_parameters(loss, credit_score, result)
        new_weight = self.weight + adjustment_weight
        new_bias = self.bias + adjustment_bias
        data = IterationData(
            prediction=prediction,
            adjustment=adjustment_weight,
            weight=self.weight,
            new_weight=new_weight,
            bias=0.0,
            new_bias=0.0
        )
        self.weight = new_weight
        return data

    def predict(self, credit_score: float) -> float:
        return credit_score * self.weight + self.bias  # Linear prediction

    def compare(self, prediction: float, result: float) -> float:
        return (result - prediction) ** 2  # Mean Squared Error

    def adjust_parameters(self, loss: float, credit_score: float, result: float) -> tuple:
        gradient = 2 * (self.predict(credit_score) - result)  # Derivative of MSE
        adjustment_weight = -gradient * credit_score * self.learning_rate
        adjustment_bias = -gradient * self.learning_rate
        return adjustment_weight, adjustment_bias