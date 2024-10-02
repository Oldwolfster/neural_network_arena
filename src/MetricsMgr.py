from src.Metrics import Metrics
from Utils import EpochSummary
from typing import List
from ConvergenceDetector import ConvergenceDetector

class MetricsMgr:
    def __init__(self, name, sample_count: int,  conv_e, conv_t, acc_t, data):
        # Run Level members
        self.name = name
        self.epoch_summaries = []           # Store the epoch summaries
        self.summary = EpochSummary()       # The current Epoch summary
        self.run_time = 0                   # How long did training take?
        self.iteration_num = 0              # Current Iteration #
        self.sample_count = sample_count    # Number of samples in each iteration.
        self.accuracy_threshold = acc_t     # In regression, how close must it be to be considered "accurate"
        Metrics.set_acc_threshold(acc_t)    # Set at Class level (not instance) one value shared across all instances
        self.epoch_curr_number = 1          # Which epoch are we currently on.
        self.metrics = []                   # The list of metrics this manager is running.
        self.converge_detector              = ConvergenceDetector(conv_t, conv_e)
    def record_iteration(self, result):
        self.iteration_num += 1
        data = Metrics(result)
        self.metrics.append(data)
        self.update_summary(data)

    def update_summary(self, data: Metrics):
        self.summary.total_absolute_error += abs(data.prediction - data.target)
        #print(f"data.prediction = {data.prediction}\tdata.target = {data.target} ")
        self.summary.total_squared_error += (data.prediction - data.target) ** 2
        self.summary.final_weight = data.new_weight
        epsilon = 1e-6  # Small tolerance for floating-point comparisons

        if data.target == 0:  # True class is Negative
            if abs(data.prediction) < epsilon:
                # Model predicts Negative (approximately zero), true class is Negative
                self.summary.tn += 1
            else:
                # Model predicts Positive, true class is Negative
                self.summary.fp += 1
        else:  # True class is Positive
            relative_error = abs((data.target - data.prediction) / data.target)
            if relative_error <= self.accuracy_threshold:
                # Model's prediction is within the threshold
                self.summary.tp += 1
            else:
                # Model's prediction is outside the threshold
                self.summary.fn += 1



        """
        if data.target == 0: # It's a Negative, either True  or False
            if data.target == data.prediction:
                self.summary.tn += 1
            else:
                self.summary.fn += 1
        else:  # It's a Positive True if prediction is positive or within threshold for regression
            if abs((data.target - data.prediction) / data.target) <= self.accuracy_threshold:
                #print(f"True Positive, data.target={data.target}\tdata.prediction={data.prediction}")
                self.summary.tp += 1
            else:
                #print(f"False Negative, data.target={data.target}\tdata.prediction={data.prediction}")
                self.summary.fp += 1
        """
    def finish_epoch_summary(self):
        self.summary.model_name = self.name
        self.summary.epoch = self.epoch_curr_number

        self.epoch_curr_number += 1
        self.summary.total_samples = self.sample_count
        #self.summary.total_absolute_error= self.converge_tae_current
        self.iteration_num = 0  # Reset counter back to zero
        self.epoch_summaries.append(self.summary)
        #print(f"epoch:{self.summary.epoch}self.summary.total_absolute_error = {self.summary.total_absolute_error}")
        epochs_to_remove = self.converge_detector.check_convergence(self.summary.total_absolute_error) #, self.summary.final_weight)
        if epochs_to_remove == 0:      # 0 indicates it has NOT converged
            self.summary = EpochSummary()   # Create summary for next epoch
            return False
        # Still here so it has converged
        self.remove_converged_epochs_from_logs(epochs_to_remove)
        return True

    def remove_converged_epochs_from_logs(self, epochs_to_remove):
        #print(f"epochs to remove:{epochs_to_remove}\tepochs :{epochs}")
        iterations_to_remove = epochs_to_remove * self.sample_count
        del self.epoch_summaries[-epochs_to_remove:]
        del self.metrics[-iterations_to_remove:]



