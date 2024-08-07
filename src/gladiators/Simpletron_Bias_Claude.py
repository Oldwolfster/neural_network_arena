from src.arena import *
from src.metrics import Metrics
from src.gladiator import Gladiator

class Simpletron_Bias_Claude(Gladiator):
    def __init__(self, number_of_epochs, metrics):
        super().__init__(number_of_epochs)
        self.metrics = metrics
        self.weight = 0.2  # Initial weight
        self.bias = 0  # Initial bias
        self.learning_rate = 0.001

    def train(self, training_data):
        for epoch in range(self.number_of_epochs):
            self.run_a_epoch(training_data, epoch)

    def run_a_epoch(self, train_data, epoch_num):
        for i, (credit_score, result) in enumerate(train_data):
            self.training_iteration(i, epoch_num, credit_score, result)
        self.metrics.epoch_completed()

    def training_iteration(self, i, epoch, credit_score, result):
        prediction = self.predict(credit_score)
        loss = self.compare(prediction, result)
        self.adjust_parameters(loss, credit_score)
        self.metrics.record_iteration_metrics(i, epoch, credit_score, result, prediction, loss,
                                              self.weight - self.weight, self.weight, self.weight, self.metrics)

    def predict(self, credit_score):
        return 1 if credit_score * self.weight + self.bias >= 0.5 else 0

    def compare(self, prediction, result):
        return result - prediction

    def adjust_parameters(self, loss, credit_score):
        # Adjust weight
        weight_adjustment = loss * credit_score * self.learning_rate
        self.weight += weight_adjustment

        # Adjust bias
        bias_adjustment = loss * self.learning_rate
        self.bias += bias_adjustment



