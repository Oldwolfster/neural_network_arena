from src.arena import *
from src.metrics import Metrics
from src.gladiator import Gladiator

class Template_Simpletron_ChatGPT(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.

    In order to utilize the metrics there are two steps required:
    1) After each iteration call metrics.record_iteration_metrics
    2) After each iteration call metrics.record_epoch_metrics (no additional info req - uses info from iterations)
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, learning_rate: float = 0.001, initial_weight: float = 0.2):
        super().__init__(number_of_epochs)
        self.metrics = metrics
        self.weight = initial_weight
        self.learning_rate = learning_rate

    def train(self, training_data):
        for epoch in range(self.number_of_epochs):
            if self.run_an_epoch(training_data, epoch):
                break

    def run_an_epoch(self, train_data, epoch_num: int) -> bool:
        for i, (credit_score, result) in enumerate(train_data):
            self.training_iteration(i, epoch_num, credit_score, result)
        return self.metrics.epoch_completed()

    def training_iteration(self, i: int, epoch: int, credit_score: float, result: int) -> None:
        prediction = self.predict(credit_score)
        loss = self.compare(prediction, result)
        adjustment = self.adjust_weight(loss)
        new_weight = self.weight + adjustment
        self.metrics.record_iteration_metrics(i, epoch, credit_score, result, prediction, loss, adjustment, self.weight, new_weight, self.metrics)
        self.weight = new_weight

    def predict(self, credit_score: float) -> int:
        return 1 if round(credit_score * self.weight, 7) >= 0.5 else 0

    def compare(self, prediction: int, result: int) -> float:
        return result - prediction

    def adjust_weight(self, loss: float) -> float:
        return loss * self.learning_rate
