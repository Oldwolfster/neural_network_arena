from Arena import *
import numpy as np


class SimpletronWithL1L2Regularization(Gladiator):
    def __init__(self, number_of_epochs, metrics):
        super().__init__(number_of_epochs)
        self.metrics = metrics
        self.neuron_weight = 0.5  # Initial weight
        self.learning_rate = 0.02  # Learning rate
        self.l1_lambda = 0.01  # L1 regularization strength
        self.l2_lambda = 0.01  # L2 regularization strength

    def train(self, training_data):
        for epoch in range(self.number_of_epochs):
            self.run_epoch(training_data, epoch)

    def run_epoch(self, train_data, epoch_num):
        self.metrics.clear_epoch_level()
        for i, (credit_score, result) in enumerate(train_data):
            self.training_iteration(i, credit_score, result)

    def training_iteration(self, i, credit_score, result):
        prediction = self.predict(credit_score)
        loss = self.compare(prediction, result)
        adjustment = self.adjust_weight(loss, credit_score)
        new_weight = self.neuron_weight + adjustment
        self.record_iteration_metrics(i, credit_score, result, prediction, loss, adjustment, self.neuron_weight, new_weight)
        self.neuron_weight = new_weight

    def predict(self, credit_score):
        return 1 if credit_score * self.neuron_weight >= 0.5 else 0

    def compare(self, prediction, result):
        return result - prediction

    def adjust_weight(self, loss, score):
        gradient = loss * score
        l1_grad = self.l1_lambda * np.sign(self.neuron_weight)
        l2_grad = self.l2_lambda * self.neuron_weight
        total_gradient = gradient + l1_grad + l2_grad
        return -self.learning_rate * total_gradient

    def calculate_total_loss(self, prediction, result):
        mse_loss = (prediction - result) ** 2
        l1_loss = self.l1_lambda * np.abs(self.neuron_weight)
        l2_loss = 0.5 * self.l2_lambda * (self.neuron_weight ** 2)
        return mse_loss + l1_loss + l2_loss

    def record_iteration_metrics(self, i, credit_score, result, prediction, loss, adjustment, weight, new_weight):
        self.metrics.predictions.append(prediction)
        self.metrics.targets.append(result)
        total_loss = self.calculate_total_loss(prediction, result)
        self.metrics.total_loss_for_epoch += total_loss
        log_entry = (
            f"Trn Data #{i + 1}:\tNeuron Weight:\t{weight:.3f}\t\tCredit Score:\t{credit_score:.3f}\t"
            f"{'Did Repay:  ' if result == 1 else 'Did Default:'}\t{result:.3f}\t"
            f"Predicted {'CORRECTLY' if loss == 0 else 'WRONG'}\n"
            f"Predict:\t\t{weight:.2f} * {credit_score:.2f} =\t{(weight * credit_score):.3f}\t\t"
            f"if {(weight * credit_score):.3f} > .5\tThen \t{'Will Repay:  ' if prediction == 1 else 'Will Default:'}\t{prediction:.3f}\n"
            f"Compare:\t\tResult:\t\t\t{result:.3f}\t\tPrediction:\t\t{prediction:.3f}\tLoss:\t\t\t{loss:.3f}\n"
            f"Adjust:\t\t\tOrig Weight:\t{weight:.3f}\t\tAdjustment:\t{adjustment:.3f}\tNew Weight:\t\t{new_weight:.3f}\n"
        )
        self.metrics.log_list.append(log_entry)
        self.metrics.weights_this_epoch.append(self.neuron_weight)