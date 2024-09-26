from src.Metrics import Metrics
from Utils import EpochSummary
import math
from typing import List, Dict
from collections import defaultdict


class MetricsMgr:
    def __init__(self, name, conv_e, conv_t, acc_t, data):
        # Run Level members
        self.name = name
        self.run_time = 0                   # How long did training take?
        self.iteration_count = 0            # Count number of iterations in epoch
        self.prob_type = "Binary Decision"
        self.accuracy_threshold = acc_t     # In regression, how close must it be to be considered "accurate"
        self.epoch_curr_number = 1          # Which epoch are we currently on.
        Metrics.set_acc_threshold(acc_t)    # Set at Class level (not instance) one value shared across all instances
        self.converge_epoch = 0             # Which epoch did it converge on.
        self.converge_required = conv_e     # How many epochs of no change are required before we call it convg
        self.converge_progress = 0          # Used to track how many epochs loss remains unchanged.
        self.converge_threshold = conv_t    # How close does the MAE have to be between epochs to consider it converged
        self.converge_prior_mae = 0         # This is the prior loss we are checking for stability against and if unchanged for n epochs then we call it converged.
        self.converge_mae_final=0           # The MAE the training session converged at.
        self.converge_tae_current = 0       # Sum the Total Absolute Error as we log each iteration
        self.metrics = []                   # The list of metrics this manager is running.

        # I believe these are all unnecessary as i have the data in my metrics
        self.errors = []                    # -- List of total "Absolute Error" for each epochs
        self.errors_squared = []            # -- List of total squared "Absolute Error" total for each epochs
        self.log_list = []                  # List of multiline string with details of all iteration for all epochs
        #self.log_data = []                  # List of same log data but just the data
        #self.epoch_summaries = []           # Stores the data for the "Epoch Summary Report"
        #Epoch level members
        #self.epoch_is_over = 0              # Used to reset metrics when next epoch begins (but keep values for last epoch to run)
        #self.weights_this_epoch = []        # For the current  epoch, list of weights at each iteration.
        #self.bias_values_this_epoch = []    # store the bias of each iteration
        self.predictions = []               # List of the result the model predicted for the current epoch
        self.targets = []                   # List of the actual result in the training data for the current epoch
        self.total_pap = 0                  # Sum the PAP values to display total for the epoch

    def record_iteration(self, result):
        self.iteration_count += 1
        data = Metrics(result)
#        if self.epoch_is_over == 1:
#            self.epoch_reset(data.epoch)
        self.converge_tae_current += abs(data.target - data.prediction)
        self.metrics.append(data)

    def iteration_log(self) -> List[List]:
        return [metric.to_list() for metric in self.metrics]

    def check_for_convergence(self):
        if self.converge_epoch != 0:  # Prevent from running after convergence has been detected
            return False
        mae = self.converge_tae_current / self.iteration_count
        if mae == 0:
            mae = 1E-20  # Prevent divide by zero errors

        difference_between_epochs = abs(self.converge_prior_mae - mae) / mae
        if difference_between_epochs < self.converge_threshold:  # Epoch MAE is within threshold of prior epoch
            self.converge_progress += 1
            if self.converge_progress > self.converge_required:
                self.converge_epoch = self.epoch_curr_number - self.converge_required - 1
                self.converge_mae_final = mae
                self.remove_logs_after_convergence()
                return True
        else:
            self.converge_progress = 0
            self.converge_prior_mae = mae
        return False

    def remove_logs_after_convergence(self):

        epochs_to_remove = self.converge_required + 1
        iterations_to_remove = epochs_to_remove * self.iteration_count
        print(f"***iteration_count ={self.iteration_count}\tepoch_curr_number={self.epoch_curr_number}\tepoch_curr_number={iterations_to_remove}")
        #del self.epoch_summaries[-epochs_to_remove:]
        #del self.log_data[-iterations_to_remove:]
        #del self.metrics[-iterations_to_remove:]


    def get_epoch_summaries(self) -> List[EpochSummary]:
        summaries = []
        epoch_data = defaultdict(lambda: {
            'total_samples': 0,
            'correct_predictions': 0,
            'total_absolute_error': 0.0,
            'total_squared_error': 0.0
        })

        for metric in self.metrics:
            epoch = metric.epoch
            epoch_data[epoch]['total_samples'] += 1
            epoch_data[epoch]['correct_predictions'] += int(metric.is_correct)
            epoch_data[epoch]['total_absolute_error'] += metric.absolute_error
            epoch_data[epoch]['total_squared_error'] += metric.squared_error

        for epoch, data in epoch_data.items():
            summaries.append(EpochSummary(
                model_name=self.name,
                epoch=epoch,
                total_samples=data['total_samples'],
                correct_predictions=data['correct_predictions'],
                total_absolute_error=data['total_absolute_error'],
                total_squared_error=data['total_squared_error']
            ))

        return summaries

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


def get_all_epoch_summaries(mgr_list: List[MetricsMgr]) -> Dict[str, List[EpochSummary]]:
    all_summaries = {}
    for mgr in mgr_list:
        all_summaries[mgr.name] = mgr.get_epoch_summaries()
    return all_summaries