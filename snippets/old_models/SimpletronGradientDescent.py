from Arena import *

############################################################
# Model Parameters are set here as global variables.       #
############################################################
neuron_weight   = .5        # Any value works as the training data will adjust it
learning_rate   = .02       # Reduces impact of adjustment to avoid overshooting


class SimpletronGradientDescent(Gladiator):

    def __init__(self, number_of_epochs, metrics ):
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
        prediction = self.predict(credit_score, weight)
        loss = self.compare(prediction, result)
        gradient = self.calculate_gradient(loss, credit_score)
        adjustment = self.adjust_weight(gradient, learning_rate)
        new_weight = weight + adjustment
        self.record_iteration_metrics(i, credit_score, result, prediction, loss, adjustment, weight, new_weight,
                                      self.metrics)
        return new_weight

    def predict(self, credit_score, weight):
        return 1 if credit_score * weight >= 0.5 else 0

    def compare(self, prediction, result):
        return result - prediction

    def calculate_gradient(self, loss, credit_score):
        return loss * credit_score

    def adjust_weight(self, gradient, learning_rate):
        return gradient * learning_rate


    def record_iteration_metrics(self, i, credit_score, result, prediction, loss, adjustment, weight, new_weight, metrics):
        self.metrics.predictions.append(prediction)
        self.metrics.actuals.append(result)

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
        metrics.losses.append(metrics.total_loss_for_epoch)
        metrics.weights_last_epoch = metrics.weights_this_epoch[:]