from src.arena import *
from src.metrics import Metrics
from src.gladiator import Gladiator


class Simpletron_Orig(Gladiator):

    def __init__(self, number_of_epochs, metrics):
        super().__init__(number_of_epochs)
        self.metrics = metrics
        self.weight = 0.2  # Initial weight
        self.learning_rate = 0.001

    def train(self, training_data):
        for epoch in range(self.number_of_epochs):
            self.run_a_epoch(training_data, epoch)

    def run_a_epoch(self, train_data, epoch_num):
        for i, (credit_score, result) in enumerate(train_data):
            self.training_iteration(i, epoch_num, credit_score, result)
        self.metrics.record_epoch()

    def training_iteration(self, i, epoch, credit_score, result):
        prediction  = self.predict(credit_score, self.weight)
        loss        = self.compare(prediction, result)
        adjustment  = self.adjust_weight(loss, self.learning_rate)
        new_weight  = self.weight + adjustment
        self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, adjustment, self.weight, new_weight, self.metrics)
        self.weight = new_weight

    def predict(self, credit_score, weight):        # print(f"Credit Score: {credit_score}, Weight: {weight}, Product: {product}, Result: {result}")
        return 1 if round(credit_score * weight, 7) >= 0.5 else 0  # NOTE: Credit score of 50 was incorrectly  predicting "no pay" due to fp precision.. Credit Score: 50, Weight: 0.009999999999999842, Product: 0.4999999999999921, Result: 0

    def compare(self, prediction, result):
        return result - prediction

    def adjust_weight(self, loss, learning_rate):
        return loss * learning_rate