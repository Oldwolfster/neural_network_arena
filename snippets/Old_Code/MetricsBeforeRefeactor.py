from dataclasses import dataclass

"""
A class to record and manage metrics for neural network training.

Usage:
1) Instantiate - does all prep needed
2) Each iteration call record_iteration_metrics()
3) At the end of each epoch call epoch_completed()
-- All metrics will be calculated and available
Option to improve performance when it gets slow.  use numpy array instead of list for predictions and actuals
"""


@dataclass
class IterationData:
    iteration: int
    epoch: int
    input: float
    target: float
    prediction: float
    adjustment: float
    weight: float
    new_weight: float
    bias: float = 0
    old_bias: float = 0


def ape(data):
    return abs((data.prediction - data.target) / data.target)


def pap(data):
    """
    PAP is a custom metric, prediction accuracy percentage
    It's used as an iteration to gauge the accuarcy of a prediction
    For example, if the prediction is 99 and the target is 100, PAP = 99
    """
    return (data.target-abs(data.target - data.prediction))/data.target * 100


class Metrics:
    def __init__(self, name):
        ##### Does not apply to regression -- self.percents = []                  # Overall accuracy of entire epoch as a percentage (101 is the goal!)

        # Run Level members
        self.name = name
        self.epochs_with_no_change = 0  # Used to track how many epochs loss remains unchanged.
        self.converge_required = 10         # How many epochs of no change before we call it converged?
        self.epochs_to_converge = 0         # How many epochs to converge - if it doesn't converge then it will be zero
        self.MAE_at_convergence = 0
        self.run_time = 0                   # How long did training take?


        #Epoch level members
        self.epoch_is_over = 0              # USed to reset metrics when next epoch begins (but keep values for last epoch to run)
        self.total_loss_for_epoch = 0       # sum of absolute value of the loss for all iterations in an epoch
        self.total_squared_error = 0        # sum of total squared error for calculating mse
        self.total_loss_for_last_epoch = 0  # Used to test for convergence
        self.anomalies = 0                  # Anomaly is defined as outcome that is not the statistically most likely.. At the moment we can only detect this when we generate random data.
        self.losses = []                    # List of total loss for each of epoch.
        self.squared_errors = []            # List of squared errors summed for all iterations of an epoch
        self.mseForEpoch = []
        self.iteration_count = 0            # Count number of iterations in epoch

        self.weights_this_epoch = []  # For the current  epoch, list of weights at each iteration.
        self.log_list = []                  # List of multiline string with details of the iteration
        self.predictions = []               # List of the result the model predicted.
        self.targets = []                   # List of the actual result in the training data
        self.epoch_curr_number = 0          # Which epoch are we currently on.

    def record_iteration(self, data):
        if self.epoch_is_over == 1:
            self.epoch_reset()

        self.iteration_count += 1
        self.predictions.append(data.prediction)
        self.targets.append(data.target)
        # print(f"Appending Result {result} and Prediction {prediction}")
        self.total_loss_for_epoch += self.error(data)  #rename to error
        self.total_squared_error += self.loss_mse(data)
        log_entry = f"{data.epoch + 1}:{data.iteration + 1}\tInput: {data.input:2}\tTarget: {data.target}\tPrediction: {data.prediction}\t{pap(data)}\tmse_loss: {self.loss_mse(data):.2f}\tOld Weight: {data.weight:.5f}\tNew Weight: {data.new_weight:.5f}\tOld Bias: "
        self.log_list.append(log_entry)
        self.weights_this_epoch.append(data.new_weight)
        self.epoch_curr_number = data.epoch

    def record_epoch(self):
        ##### Does not apply to regression -- self.percents.append(self.accuracy * 100)
        self.epoch_is_over = 1  # signal to clear if another epoch begins (but retain last epoch values if it doesn't
        self.losses.append(self.total_loss_for_epoch)
        self.squared_errors.append(self.total_squared_error)
        self.mseForEpoch.append(self.total_squared_error/self.iteration_count)
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
        self.iteration_count = 0
        self.weights_this_epoch.clear()
        self.total_loss_for_epoch = 0   # Reset it for the next epoch
        self.predictions.clear()
        self.targets.clear()

    def error(self,data):
        return abs(data.prediction - data.target)

    def loss_mse(self, data):
        """Loss for Mean Square Average"""
        return self.error(data) ** 2

    @property
    def tp(self) -> int:
        """True Positives - Correctly predicted when the answer is yes"""
        return sum(1 for p, a in zip(self.predictions, self.targets) if p == 1 and a == 1)
