from src.arena import *
from src.metrics import Metrics
from src.gladiator import Gladiator

class _Template_Simpletron(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.

    In order to utilize the metrics there are two steps required:
    1) After each iteration call metrics.record_iteration_metrics
    2) After each iteration call metrics.record_epoch_metrics (no additional info req - uses info from iterations)
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, *args, **kwargs):
        super().__init__(number_of_epochs, metrics, *args, **kwargs)
        # Ideally avoid overriding these, but specific models may need, so must be free to do so
        # It keeps comparisons straight if respected
        # self.weight = override_weight
        # self.learning_rate = override_learning_rate

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
        adjustment = self.adjust_weight(loss)
        new_weight = self.weight + adjustment
        self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, adjustment, self.weight, new_weight, self.metrics)
        self.weight = new_weight

    def predict(self, credit_score: float) -> int:
        return 1 if round(credit_score * self.weight, 7) >= 0.5 else 0

    def compare(self, prediction: int, result: int) -> float:
        return result - prediction

    def adjust_weight(self, loss: float) -> float:
        return loss * self.learning_rate
