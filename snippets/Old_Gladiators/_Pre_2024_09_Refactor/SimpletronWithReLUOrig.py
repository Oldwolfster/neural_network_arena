from Arena import *
# Performs better than Simpletron on Quadratic data.
############################################################
# Model Parameters are set here as global variables.       #
############################################################
neuron_weight   = .2        # Any value works as the training data will adjust it
learning_rate   = .1       # Reduces impact of adjustment to avoid overshooting


class SimpletronWithReLU(Gladiator):

    def __init__(self, number_of_epochs, metrics):
        super().__init__(number_of_epochs)
        self.metrics = metrics

    def train(self, training_data):
        global neuron_weight
        for epoch in range(self.number_of_epochs):
            neuron_weight = self.run_a_epoch(training_data, neuron_weight, epoch)

    def run_a_epoch(self, train_data, weight, epoch_num):
        self.metrics.clear_epoch_level()
        for i, (credit_score, result) in enumerate(train_data):
            weight = self.training_iteration(i, credit_score, result, weight)
        return weight

    def training_iteration(self, i, credit_score, result, weight):
        prediction  = self.predict(credit_score, weight)
        loss        = self.compare(prediction, result)
        adjustment  = self.adjust_weight(loss, credit_score, learning_rate)
        new_weight  = weight + adjustment
        self.record_iteration_metrics(i, credit_score, result, prediction, loss, adjustment, weight, new_weight, self.metrics)
        return new_weight

    def predict(self, credit_score, weight):
        # Apply ReLU activation function
        raw_output = credit_score * weight
        return max(0, raw_output)  # ReLU function

    def compare(self, prediction, result):
        return result - prediction  # This remains the same

    def adjust_weight(self, loss, score, learning_rate):
        return loss * learning_rate

    def record_iteration_metrics(self, i, credit_score, result, prediction, loss, adjustment, weight, new_weight, metrics):
        self.metrics.predictions.append(prediction)
        self.metrics.targets.append(result)
        self.metrics.total_loss_for_epoch += abs(loss)
        log_entry = (
            f"Trn Data #{i + 1}:\tNeuron Weight:\t{weight:.3f}\t\tCredit Score:\t{credit_score:.3f}\t"
            f"{'Did Repay:  ' if result == 1 else 'Did Default:'}\t{result:.3f}\t"
            f"Predicted {'CORRECTLY' if loss == 0 else 'WRONG'}\n"
            f"Predict:\t\t{weight:.2f} * {credit_score:.2f} =\t{(weight * credit_score):.3f}\t\t"
            f"if {(weight * credit_score):.3f} > .5\tThen \t{'Will Repay:  ' if prediction == 1 else 'Will Default:'}\t{prediction:.3f}\n"
            f"Compare:\t\tResult:\t\t\t{result:.3f}\t\tPrediction:\t\t{prediction:.3f}\tLoss:\t\t\t{loss:.3f}\n"
            f"Adjust:\t\t\tOrig Weight:\t{weight:.3f}\t\tLoss * Rate:\t{adjustment:.3f}\tNew Weight:\t\t{new_weight:.3f}\n"
        )
        metrics.log_list.append(log_entry)
        metrics.weights_this_epoch.append(neuron_weight)
