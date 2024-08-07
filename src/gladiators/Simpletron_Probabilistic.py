import random

# Model Parameters
neuron_weight = 10        # Initial weight
learning_rate = 0.001     # Learning rate

class Simpletron_Probabilistic(Gladiator):
    def __init__(self, number_of_epochs, metrics):
        super().__init__(number_of_epochs)
        self.metrics = metrics

    def train(self, training_data):
        global neuron_weight
        for epoch in range(self.number_of_epochs):
            neuron_weight = self.run_a_epoch(training_data, neuron_weight, epoch)

    def run_a_epoch(self, train_data, weight, epoch_num):
        for i, (credit_score, result) in enumerate(train_data):
            weight = self.training_iteration(i, epoch_num, credit_score, result, weight)
        self.metrics.epoch_completed()
        return weight

    def training_iteration(self, i, epoch, credit_score, result, weight):
        prediction = self.predict(credit_score, weight)
        loss = self.compare(prediction, result)
        adjustment = self.adjust_weight(loss, credit_score, learning_rate)
        new_weight = weight + adjustment
        self.metrics.record_iteration_metrics(i, epoch, credit_score, result, prediction, loss, adjustment, weight, new_weight, self.metrics)
        return new_weight

    def predict(self, credit_score, weight):
        # Use sigmoid function to get probability-like output
        return 1 / (1 + math.exp(-credit_score * weight))

    def compare(self, prediction, result):
        return result - prediction  # This remains the same

    def adjust_weight(self, loss, score, learning_rate):
        return loss * learning_rate * score  # Include score in weight adjustment