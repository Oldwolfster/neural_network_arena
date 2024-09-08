from src.Arena import *
from src.Metrics import Metrics
from src.Gladiator import Gladiator


class Simpletron_Bias_Claude3(Gladiator):
    """
    A simple perceptron implementation with independently adjusted bias.
    This class serves as a template for more complex implementations.
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, *args):
        super().__init__(number_of_epochs, metrics, *args)
        self.bias = 0.0  # Initialize bias to 0
        self.bias_learning_rate = self.learning_rate / 10  # Slower learning rate for bias

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

        weight_adjustment = self.adjust_weight(loss)
        bias_adjustment = self.adjust_bias(loss, credit_score)

        new_weight = self.weight + weight_adjustment
        new_bias = self.bias + bias_adjustment

        self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, weight_adjustment, self.weight,
                                      new_weight, self.metrics)

        self.weight = new_weight
        self.bias = new_bias

    def predict(self, credit_score: float) -> int:
        return 1 if round(credit_score * self.weight + self.bias, 7) >= 0.5 else 0

    def compare(self, prediction: int, result: int) -> float:
        return result - prediction

    def adjust_weight(self, loss: float) -> float:
        return loss * self.learning_rate  # This remains unchanged as per your request

    def adjust_bias(self, loss: float, credit_score: float) -> float:
        # Adjust bias based on both the loss and how far the input is from the decision boundary
        decision_distance = abs(0.5 - (credit_score * self.weight + self.bias)) #.5 because that is the cutoff for normally paid or normally not paid.
        return loss * self.bias_learning_rate * decision_distance

    # this is the most accurate model but it still seems like it