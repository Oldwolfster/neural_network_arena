from src.arena import *
from src.gladiator import Gladiator

# Model Parameters are set here as global variables.
neuron_weight = 0.2
neuron_bias = 0.0
learning_rate = .001

class Simpletron_With_Bias(Gladiator):
    def __init__(self, number_of_epochs, metrics):
        super().__init__(number_of_epochs)
        self.metrics = metrics

    def train(self, training_data):
        global neuron_weight, neuron_bias
        for epoch in range(self.number_of_epochs):
            neuron_weight, neuron_bias = self.run_a_epoch(training_data, neuron_weight, neuron_bias, epoch)

    def run_a_epoch(self, train_data, weight, bias, epoch_num):
        for i, (credit_score, result) in enumerate(train_data):
            weight, bias = self.training_iteration(i, credit_score, result, weight, bias)
        self.metrics.record_epoch()
        return weight, bias

    def training_iteration(self, i, credit_score, result, weight, bias):
        prediction = self.predict(credit_score, weight, bias)
        loss = self.compare(prediction, result)
        weight_adjustment, bias_adjustment = self.adjust_parameters(loss, credit_score, learning_rate)
        new_weight = weight + weight_adjustment
        new_bias = bias + bias_adjustment
        self.record_iteration_metrics(i, credit_score, result, prediction, loss, weight_adjustment, bias_adjustment, weight, bias, new_weight, new_bias, self.metrics)
        return new_weight, new_bias

    def predict(self, credit_score, weight, bias):
        return 1 if credit_score * weight + bias >= 0 else 0

    def compare(self, prediction, result):
        return result - prediction

    def adjust_parameters(self, loss, score, learning_rate):
        weight_adjustment = loss * score * learning_rate
        bias_adjustment = loss * learning_rate
        return weight_adjustment, bias_adjustment

    def record_iteration_metrics(self, i, credit_score, result, prediction, loss, weight_adjustment, bias_adjustment, weight, bias, new_weight, new_bias, metrics):
        self.metrics.predictions.append(prediction)
        self.metrics.actuals.append(result)
        self.metrics.total_loss_for_epoch += abs(loss)
        log_entry = (
            f"Trn Data #{i + 1}:\tWeight: {weight:.3f}\tBias: {bias:.3f}\tCredit Score: {credit_score:.3f}\t"
            f"{'Did Repay:  ' if result == 1 else 'Did Default:'} {result:.3f}\t"
            f"Predicted {'CORRECTLY' if loss == 0 else 'WRONG'}\n"
            f"Predict:\t{weight:.2f} * {credit_score:.2f} + {bias:.2f} = {(weight * credit_score + bias):.3f}\t"
            f"if {(weight * credit_score + bias):.3f} >= 0\tThen {'Will Repay:  ' if prediction == 1 else 'Will Default:'} {prediction:.3f}\n"
            f"Compare:\tResult: {result:.3f}\tPrediction: {prediction:.3f}\tLoss: {loss:.3f}\n"
            f"Adjust:\tOrig Weight: {weight:.3f}\tWeight Adj: {weight_adjustment:.3f}\tNew Weight: {new_weight:.3f}\n"
            f"\tOrig Bias: {bias:.3f}\tBias Adj: {bias_adjustment:.3f}\tNew Bias: {new_bias:.3f}\n"
        )
        metrics.log_list.append(log_entry)
        metrics.weights_this_epoch.append(neuron_weight)