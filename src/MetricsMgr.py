from src.Metrics import Metrics, GladiatorOutput, IterationResult, IterationContext
import math


class MetricsMgr:
    def __init__(self, name, conv_e, conv_t, acc_t, data):

        # NEW PROPS Run Level members
        self.metrics = []  # The list of metrics this manager is running.

        # Run Level members
        self.name = name

        self.prob_type = "Binary Decision"
        self.accuracy_threshold = acc_t     # In regression, how close must it be to be considered "accurate"
        self.epoch_curr_number = 1          # Which epoch are we currently on.
        self.converge_epoch = 0             # Which epoch did it converge on.
        self.converge_required = conv_e     # How many epochs of no change are required before we call it convg
        self.converge_progress = 0          # Used to track how many epochs loss remains unchanged.
        self.converge_threshold = conv_t    # How close does the MAE have to be between epochs to consider it converged
        self.converge_mae = 0               # This is the prior loss we are checking for stability against and if unchanged for n epochs then we call it converged.
        self.converge_mae_final=0           # The MAE the training session converged at.
        self.run_time = 0                   # How long did training take?
        self.metrics = []                   # The list of metrics this manager is running.
        self.errors = []                    # -- List of total "Absolute Error" for each epochs
        self.errors_squared = []            # -- List of total squared "Absolute Error" total for each epochs
        self.log_list = []                  # List of multiline string with details of all iteration for all epochs
        self.log_data = []                  # List of same log data but just the data
        self.epoch_summaries = []           # Stores the data for the "Epoch Summary Report"
        self.iteration_count = 0            # Count number of iterations in epoch

        #Epoch level members
        self.epoch_is_over = 0              # Used to reset metrics when next epoch begins (but keep values for last epoch to run)
        self.weights_this_epoch = []        # For the current  epoch, list of weights at each iteration.
        self.bias_values_this_epoch = []    # store the bias of each iteration
        self.predictions = []               # List of the result the model predicted for the current epoch
        self.targets = []                   # List of the actual result in the training data for the current epoch
        self.total_pap = 0                  # Sum the PAP values to display total for the epoch

    def record_iteration(self, result):
        if self.epoch_is_over == 1:
            self.epoch_reset(data.epoch)
        self.iteration_count += 1
        data = Metrics(result)
        self.metrics.append(data)
        print([   f"{data.epoch}",
                  f"{data.iteration}",
                  f"{data.input:,.2f}",
                  f"{data.target:,.2f}",
                  f"{data.prediction:,.2f}",
                  # (t - p) / (t + 1e-64) BELOW IS TEMP TO CHECK ACCURACY WITH REGRESSIONM                      f"{abs((data.target - data.prediction)  /(data.target + 1e-64)) :,.4f}",
                  f"{abs(data.target - data.prediction):,.2f}",  # Error calculation
                  f"{data.weight:,.4f}",
                  f"{data.new_weight:,.4f}",
                  f"{data.bias:,.4f}",
                  f"{data.new_bias:,.4f}"])

    def record_iteration_ORIG(self,     data):
        if self.epoch_is_over == 1:
            self.epoch_reset(data.epoch)
        self.iteration_count += 1
        self.predictions.append(data.prediction)
        self.targets.append(data.target)

        # OLd logic for text log log_entry = f"{data.epoch }:{data.iteration }\tInput: {data.input:,.2f}\tTarget: {data.target:,.2f}\tPrediction: {data.prediction:,.2f}\t{self.pap(data):,.0f}\terror: {self.error(data):.2f}\tOld Weight: {data.weight:,.2f}\tNew Weight: {data.new_weight:,.2f}"
        # OLd logic for text log self.log_list.append(log_entry)
        print(f"data.iteration: {data.iteration}:\tinput{data.input}\tnew weight{data.new_weight}")
        self.log_data.append([f"{data.epoch}",
                         f"{data.iteration}",
                         f"{data.input:,.2f}",
                         f"{data.target:,.2f}",
                         f"{data.prediction:,.2f}",   # (t - p) / (t + 1e-64) BELOW IS TEMP TO CHECK ACCURACY WITH REGRESSIONM                      f"{abs((data.target - data.prediction)  /(data.target + 1e-64)) :,.4f}",
                         f"{abs(data.target - data.prediction):,.2f}",  # Error calculation
                         f"{data.weight:,.4f}",
                         f"{data.new_weight:,.4f}",
                         f"{data.bias:,.4f}",
                         f"{data.new_bias:,.4f}"])

        #self.log_data.append( [data.epoch, data.iteration,data.input, data.target,data.prediction, self.error(data),data.weight,data.new_weight ])
        self.weights_this_epoch.append(data.new_weight)
        self.bias_values_this_epoch.append(data.new_bias)

    @property
    def tp(self) -> int:
        return sum(m.is_true_positive for m in self.metrics)

    @property
    def tn(self) -> int:
        return sum(m.is_true_negative for m in self.metrics)

    @property
    def fp(self) -> int:
        return sum(m.is_false_positive for m in self.metrics)

    @property
    def fn(self) -> int:
        return sum(m.is_false_negative for m in self.metrics)

    @property
    def correct(self) -> int:
        return self.tp + self.tn

    @property
    def wrong(self) -> int:
        return self.fp + self.fn

    @property
    def accuracy(self) -> float:
        total = len(self.metrics)
        return self.correct / total if total > 0 else 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0

    @property
    def f1(self) -> float:
        return 2 * (self.precision * self.recall) / (self.precision + self.recall) if (
                                                                                                  self.precision + self.recall) > 0 else 0

    @property
    def mean_absolute_error(self) -> float:
        return sum(m.absolute_error for m in self.metrics) / len(self.metrics) if self.metrics else 0

    @property
    def total_absolute_error(self) -> float:
        return sum(m.absolute_error for m in self.metrics)

    @property
    def mean_squared_error(self) -> float:
        return sum(m.squared_error for m in self.metrics) / len(self.metrics) if self.metrics else 0

    @property
    def root_mean_squared_error(self) -> float:
        return math.sqrt(self.mean_squared_error)

    @property
    def sum_squared_errors(self) -> float:
        return sum(m.squared_error for m in self.metrics)

    def summary(self) -> dict:
        return {
            "Total Samples": len(self.metrics),
            "Correct": self.correct,
            "Wrong": self.wrong,
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 Score": self.f1,
            "Mean Absolute Error": self.mean_absolute_error,
            "Total Absolute Error": self.total_absolute_error,
            "Mean Squared Error": self.mean_squared_error,
            "Root Mean Squared Error": self.root_mean_squared_error,
            "Sum of Squared Errors": self.sum_squared_errors
        }