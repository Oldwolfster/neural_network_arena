from src.arena import *

# Add metric, epochs to stabilization.

############################################################
# Model Parameters are set here as global variables.       #
############################################################
neuron_weight   = .2        # Any value works as the training data will adjust it
learning_rate   = .001       # Reduces impact of adjustment to avoid overshooting


class Simpletron(Gladiator):

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
            weight = self.training_iteration(i, epoch_num, credit_score, result, weight)
        self.record_epoch_metrics()
        return weight

    def training_iteration(self, i, epoch, credit_score, result, weight):
        prediction  = self.predict(credit_score, weight)
        loss        = self.compare(prediction, result)
        adjustment  = self.adjust_weight(loss, credit_score, learning_rate)
        new_weight  = weight + adjustment
        self.record_iteration_metrics(i, epoch, credit_score, result, prediction, loss, adjustment, weight, new_weight, self.metrics)
        return new_weight

    def predict(self, credit_score, weight):
        #return 1 if credit_score * weight >= 0.5 else 0
        #product = credit_score * weight
        #result = 1 if product >= 0.5 else 0
        #print(f"Credit Score: {credit_score}, Weight: {weight}, Product: {product}, Result: {result}")
        #return result
        # NOTE: Credit score of 50 was incorrectly  predicting "no pay" due to fp precision.. Credit Score: 50, Weight: 0.009999999999999842, Product: 0.4999999999999921, Result: 0
        return 1 if round(credit_score * weight, 7) >= 0.5 else 0


    def compare(self, prediction, result):
        return result - prediction  # This remains the same

    def adjust_weight(self, loss, score, learning_rate):
        return loss * learning_rate

    def record_epoch_metrics(self):
        self.metrics.losses.append(self.metrics.total_loss_for_epoch)   # Record the total MAE for the epoch
        self.metrics.total_loss_for_epoch = 0                           # Reset it for the next epoch

    def record_iteration_metrics(self, iteration, epoch, credit_score, result, prediction, loss, adjustment, weight, new_weight, metrics):
        self.metrics.predictions.append(prediction)
        self.metrics.actuals.append(result)
        self.metrics.total_loss_for_epoch += abs(loss)
        print(f"epoch: {epoch + 1} iteration: {iteration}\tMAE for iteration ={abs(loss)}")
        log_entry = f"epoch: {epoch+1} iteration: {iteration}\tcredit score: {credit_score:2}\tresult: {result}\tprediction: {prediction}\tloss: {loss:.2f}\tresult: {'CORRECT' if loss == 0 else 'WRONG'}\told weight: {weight:.5f}\tnew weight: {new_weight:.5f}"
        metrics.log_list.append(log_entry)
        metrics.weights_this_epoch.append(neuron_weight)


        # log_entry = (
        #     f"Trn Data #{i + 1}:\tNeuron Weight:\t{weight:.3f}\t\tCredit Score:\t{credit_score:.3f}\t"
        #    f"{'Did Repay:  ' if result == 1 else 'Did Default:'}\t{result:.3f}\t"
        #    f"Predicted {'CORRECTLY' if loss == 0 else 'WRONG'}\n"
        #    f"Predict:\t\t{weight:.2f} * {credit_score:.2f} =\t{(weight * credit_score):.3f}\t\t"
        #    f"if {(weight * credit_score):.3f} > .5\tThen \t{'Will Repay:  ' if prediction == 1 else 'Will Default:'}\t{prediction:.3f}\n"
        #    f"Compare:\t\tResult:\t\t\t{result:.3f}\t\tPrediction:\t\t{prediction:.3f}\tLoss:\t\t\t{loss:.3f}\n"
        #    f"Adjust:\t\t\tOrig Weight:\t{weight:.3f}\t\tLoss * Rate:\t{adjustment:.3f}\tNew Weight:\t\t{new_weight:.3f}\n"
        #)

