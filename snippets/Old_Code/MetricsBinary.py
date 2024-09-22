"""
A class to record and manage metrics for neural network training.

Usage:
1) Instantiate - does all prep needed
2) Each iteration call record_iteration_metrics()
3) At the end of each epoch call epoch_completed()
-- All metrics will be calculated and available
Option to improve performance when it gets slow.  use numpy array instead of list for predictions and actuals
"""


class MetricsBinary:
    def __init__(self, name):
        self.name = name
        self.epoch_is_over = 0              # USed to reset metrics when next epoch begins (but keep values for last epoch to run)
        self.total_loss_for_epoch = 0       # sum of absolute value of the loss for all iterations in an epoch
        self.total_loss_for_last_epoch = 0  # Used to test for convergence
        self.epochs_with_no_change = 0      # Used to track how many epochs loss remains unchanged.
        self.anomalies = 0                  # Anomaly is defined as outcome that is not the statistically most likely.. At the moment we can only detect this when we generate random data.
        self.losses = []                    # List of total loss for each of epoch.
        self.percents = []                  # Overall accuracy of entire epoch as a percentage (101 is the goal!)
        self.weights_this_epoch = []        # For the current  epoch, list of weights at each iteration.
        #self.weights_last_epoch = []        # For the previous epoch, list of weights at each iteration.
        self.log_list = []                  # List of multiline string with details of the iteration
        self.predictions = []               # List of the result the model predicted.
        self.actuals = []                   # List of the actual result in the training data
        self.epoch_curr_number = 0          # Which epoch are we currently on.
        self.converge_required = 10         # How many epochs of no change before we call it converged?
        self.epochs_to_converge = 0         # How many epochs to converge - if it doesn't converge then it will be zero
        self.MAE_at_convergence = 0
        self.run_time = 0                   # How long did training take?

    def record_iteration(self, iteration, epoch, input_value, result, prediction, loss, adjustment, weight, new_weight, metrics, bias=0, old_bias=0):
        if self.epoch_is_over == 1:
            self.epoch_reset()

        self.predictions.append(prediction)
        self.actuals.append(result)
        # print(f"Appending Result {result} and Prediction {prediction}")
        self.total_loss_for_epoch += abs(loss)
        #8/26/2024 log_entry = f"{epoch+1}:{iteration + 1}\tInput: {input_value:2}\tResult: {result}\tPrediction: {prediction}\tloss: {loss:.2f}\tResult: {'CORRECT' if loss == 0 else 'WRONG'}\tOld Weight: {weight:.5f}\tNew Weight: {new_weight:.5f}\tOld Bias: {weight:.5f}\tNew Bias: {new_weight:.5f}"
        log_entry = f"{epoch + 1}:{iteration + 1}\tInput: {input_value:2}\tTarget: {result}\tPrediction: {prediction}\t{'CORRECT' if loss == 0 else 'WRONG'}\tloss: {loss:.2f}\tOld Weight: {weight:.5f}\tNew Weight: {new_weight:.5f}\tOld Bias: {old_bias:.5f}\tNew Bias: {bias:.5f}"
        self.log_list.append(log_entry)
        self.weights_this_epoch.append(new_weight)
        self.epoch_curr_number = epoch

    def record_epoch(self):
        self.epoch_is_over = 1  # signal to clear if another epoch begins (but retain last epoch values if it doesn't
        self.losses.append(self.total_loss_for_epoch)
        self.percents.append(self.accuracy * 100)
        self.check_for_epoch_convergence()
        self.total_loss_for_last_epoch = self.total_loss_for_epoch  # record loss to compare after next epoch

    def check_for_epoch_convergence(self):
        # print(f'self.total_loss_for_epoch\t{self.total_loss_for_epoch}\tself.total_loss_for_last_epoch\t{self.total_loss_for_last_epoch}\tself.epochs_with_no_change\t{self.epochs_with_no_change}\t')
        if self.total_loss_for_epoch == self.total_loss_for_last_epoch:            # No change in loss, so increment convergence counter
            self.epochs_with_no_change += 1
        else:
            self.epochs_with_no_change = 0
        if self.epochs_with_no_change > self.converge_required and self.epochs_to_converge == 0:
            self.epochs_to_converge = self.epoch_curr_number - self.converge_required
            self.MAE_at_convergence = self.wrong

    def epoch_reset(self):
        """When a new epoch begins, reset values"""
        self.epoch_is_over = 0          # Reset flag
        self.weights_this_epoch.clear()
        self.total_loss_for_epoch = 0   # Reset it for the next epoch
        self.predictions.clear()
        self.actuals.clear()

    @property
    def tp(self) -> int:
        """True Positives - Correctly predicted when the answer is yes"""
        return sum(1 for p, a in zip(self.predictions, self.actuals) if p == 1 and a == 1)

    @property
    def tn(self) -> int:
        """True Negatives - Correctly predicted when the answer is no"""
        return sum(1 for p, a in zip(self.predictions, self.actuals) if p == 0 and a == 0)

    @property
    def fp(self) -> int:
        """False Positives - Model said yes, but not true aka Type 1 Error"""
        return sum(1 for p, a in zip(self.predictions, self.actuals) if p == 1 and a == 0)

    @property
    def fn(self)  -> int:
        """False Negatives - Model said no, but should have said yes..  AKA Type II Error"""
        return sum(1 for p, a in zip(self.predictions, self.actuals) if p == 0 and a == 1)

    @property
    def correct(self) -> int:
        return self.tp + self.tn        #true positive plus true negatives

    @property
    def wrong(self) -> int:
        return self.fp + self.fn

    @property
    def precision(self) -> float:
        if (self.tp + self.fp) > 0:
            return self.tp / (self.tp + self.fp)
        return 0

    @property
    def recall(self) -> float:
        if (self.tp + self.fn) > 0:
            return self.tp / (self.tp + self.fn)
        return 0

    @property
    def f1_score(self) -> float:
        precision = self.precision
        recall = self.recall
        if (precision + recall) > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        if total > 0:
            return (self.tp + self.tn) / total
        return 0

# Old log entry
# log_entry = (
#    f"Trn Data #{i + 1}:\tNeuron Weight:\t{weight:.3f}\t\tCredit Score:\t{credit_score:.3f}\t"
#    f"{'Did Repay:  ' if result == 1 else 'Did Default:'}\t{result:.3f}\t"
#    f"Predicted {'CORRECTLY' if loss == 0 else 'WRONG'}\n"
#    f"Predict:\t\t{weight:.2f} * {credit_score:.2f} =\t{(weight * credit_score):.3f}\t\t"
#    f"if {(weight * credit_score):.3f} > .5\tThen \t{'Will Repay:  ' if prediction == 1 else 'Will Default:'}\t{prediction:.3f}\n"
#    f"Compare:\t\tResult:\t\t\t{result:.3f}\t\tPrediction:\t\t{prediction:.3f}\tLoss:\t\t\t{loss:.3f}\n"
#    f"Adjust:\t\t\tOrig Weight:\t{weight:.3f}\t\tLoss * Rate:\t{adjustment:.3f}\tNew Weight:\t\t{new_weight:.3f}\n"
# )

