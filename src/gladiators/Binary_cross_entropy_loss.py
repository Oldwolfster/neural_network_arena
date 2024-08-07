import random
import math

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
        loss = self.binary_cross_entropy_loss(prediction, result)
        adjustment = self.adjust_weight(prediction, result, credit_score, learning_rate)
        new_weight = weight + adjustment
        self.metrics.record_iteration_metrics(i, epoch, credit_score, result, prediction, loss, adjustment, weight, new_weight, self.metrics)
        return new_weight

    def predict(self, credit_score, weight):
        # Use sigmoid function to get probability-like output
        return 1 / (1 + math.exp(-credit_score * weight))

    def binary_cross_entropy_loss(self, prediction, result):
        epsilon = 1e-15  # Small value to avoid log(0)
        prediction = max(min(prediction, 1 - epsilon), epsilon)  # Clip prediction to avoid log(0)
        return -result * math.log(prediction) - (1 - result) * math.log(1 - prediction)

    def adjust_weight(self, prediction, result, score, learning_rate):
        error = result - prediction
        return learning_rate * error * score

def generate_random_linear_data(include_anomalies):
    training_data = []
    for _ in range(qty_rand_data):
        score = random.randint(1, 100)
        if include_anomalies:
            second_number = 1 if random.random() < (score / 100) else 0
        else:
            second_number = 1 if score >= 50 else 0
        training_data.append((score / 100, second_number))  # Normalize score to [0, 1]
    return training_data