from src.Arena import *
from src.Metrics import Metrics, IterationData
from src.Gladiator import Gladiator

class LinearRegression(Gladiator):
    """
    A simple linear regression implementation for educational purposes.
    This class serves as a template for more complex implementations.
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, *args):
        super().__init__(number_of_epochs, metrics, *args)
        self.bias = 0.0  # Initialize bias term

    def train(self, training_data):
        for epoch in range(self.number_of_epochs):
            if self.run_an_epoch(training_data, epoch):
                break

    def run_an_epoch(self, train_data, epoch_num: int) -> bool:
        for i, (credit_score, result) in enumerate(train_data):
            self.training_iteration(i, epoch_num, credit_score, result)
        return self.metrics.record_epoch()

    def training_iteration(self, i: int, epoch: int, credit_score: float, result: float) -> None:
        prediction = self.predict(credit_score)
        loss = self.compare(prediction, result)
        adjustment_weight, adjustment_bias = self.adjust_parameters(loss, credit_score, result)
        new_weight = self.weight + adjustment_weight
        new_bias = self.bias + adjustment_bias
        data = IterationData(
            iteration=i + 1,
            epoch=epoch + 1,
            input=credit_score,
            target=result,
            prediction=prediction,
            adjustment=adjustment_weight,
            weight=self.weight,
            new_weight=new_weight,
            bias=self.bias,
            new_bias=new_bias

        )

        self.metrics.record_iteration(data)
        self.weight = new_weight
        self.bias = new_bias

    def predict(self, credit_score: float) -> float:
        return credit_score * self.weight + self.bias  # Linear prediction

    def compare(self, prediction: float, result: float) -> float:
        return (result - prediction) ** 2  # Mean Squared Error

    def adjust_parameters(self, loss: float, credit_score: float, result: float) -> tuple:
        gradient = 2 * (self.predict(credit_score) - result)  # Derivative of MSE
        adjustment_weight = -gradient * credit_score * self.learning_rate
        adjustment_bias = -gradient * self.learning_rate
        return adjustment_weight, adjustment_bias