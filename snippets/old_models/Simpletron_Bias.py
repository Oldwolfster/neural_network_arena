from src.TrainingPit import *
from src.Metrics import Metrics
from src.Gladiator import Gladiator

# Add metric, epochs to stabilization.

############################################################
# Model Parameters are set here as global variables.       #
############################################################
neuron_weight   = .2        # Any value works as the training data will adjust it
learning_rate   = .001       # Reduces impact of adjustment to avoid overshooting
neuron_bias = 0

class Simpletron_Bias(Gladiator):

    def __init__(self, number_of_epochs, metrics):
        super().__init__(number_of_epochs)
        self.metrics = metrics

    def train(self, training_data):
        global neuron_weight
        global neuron_bias
        for epoch in range(self.number_of_epochs):
            neuron_weight, neuron_bias = self.run_a_epoch(training_data, neuron_weight, epoch)

    def run_a_epoch(self, train_data, weight, epoch_num):
        for i, (credit_score, result) in enumerate(train_data):
            weight, bias = self.training_iteration(i, epoch_num, credit_score, result, weight)
        self.metrics.record_epoch()
        return weight, bias

    def training_iteration(self, i, epoch, credit_score, result, weight):
        prediction  = self.predict(credit_score, weight)
        loss        = self.compare(prediction, result)
        adjustment  = self.adjust_weight(loss, credit_score, learning_rate)
        new_weight  = weight + adjustment
        new_bias  = neuron_bias + adjustment
        self.metrics.record_iteration(i, epoch, credit_score, result, prediction, loss, adjustment, weight, new_weight, self.metrics)
        return new_weight, new_bias

    def predict(self, credit_score, weight):
        #return 1 if credit_score * weight >= 0.5 else 0
        #product = credit_score * weight
        #result = 1 if product >= 0.5 else 0
        #print(f"Credit Score: {credit_score}, Weight: {weight}, Product: {product}, Result: {result}")
        #return result
        return 1 if round(credit_score * weight + neuron_bias, 7) >= 0.5 else 0  # NOTE: Credit score of 50 was incorrectly  predicting "no pay" due to fp precision.. Credit Score: 50, Weight: 0.009999999999999842, Product: 0.4999999999999921, Result: 0

    def compare(self, prediction, result):
        return result - prediction  # This remains the same

    def adjust_weight(self, loss, score, learning_rate):
        return loss * learning_rate
