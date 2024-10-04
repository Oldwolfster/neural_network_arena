from src.Engine import *
from src.Metrics import Metrics
from src.BaseGladiator import Gladiator

class Simpletron_Bias_Perplexity(Gladiator):
    """
    A simple perceptron implementation for educational purposes, now including bias.
    This class serves as a template for more complex implementations.

    In order to utilize the metrics there are two steps required:
    1) After each iteration call metrics.record_iteration_metrics
    2) After each iteration call metrics.record_epoch_metrics (no additional info req - uses info from iterations)
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, *args):
        super().__init__(number_of_epochs, metrics, *args)
        self.bias = 0  # Initialize bias to 0

    def train(self, training_data):
        for epoch in range(self.number_of_epochs):
            if self.run_an_epoch(training_data, epoch):
                break

    def run_an_epoch(self, train_data, epoch_num: int) -> bool:
        for i, (credit_score, result) in enumerate(train_data):
            self.training_iteration(i, epoch_num, credit_score, result)
        return self.metrics.record_epoch()

    def training_iteration(self, i: int, epoch: int, credit_score: float, result: int) -> None:
        prediction = self.predict(credit_score)
        loss = self.compare(prediction, result)
        weight_adjustment, bias_adjustment = self.calculate_adjustments(loss, credit_score)
        new_weight = self.weight + weight_adjustment
        new_bias = self.bias + bias_adjustment
        self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, weight_adjustment, self.weight, new_weight, self.metrics)
        self.weight = new_weight
        self.bias = new_bias

    def predict(self, credit_score: float) -> int:
        return 1 if round(credit_score * self.weight + self.bias, 7) >= 0.5 else 0

    def compare(self, prediction: int, result: int) -> float:
        return result - prediction

    def calculate_adjustments(self, loss: float, input_value: float) -> tuple:
        weight_adjustment = loss * self.learning_rate * input_value
        bias_adjustment   = loss * self.learning_rate
        return weight_adjustment, bias_adjustment

    def adjust_weight(self, loss: float) -> float:
        # This method is kept for compatibility, but we'll use calculate_adjustments instead
        return loss * self.learning_rate