from src.Metrics import Metrics
from Utils import EpochSummary
from typing import List
from ConvergenceDetectorOrig import ConvergenceDetector

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
        self.summary.total_squared_error += (data.prediction - data.target) ** 2
        self.summary.final_weight = data.new_weight
        if data.target == 0: # It's a True Neg or False Neg
            """True Positives - Correctly predicted within threshold when target is non-zero"""
            #if abs((t - p) / (t + 1e-64)) <= self.accuracy_threshold and t != 0)
            if data.target == data.prediction:
                self.summary.tn += 1
            else:
                self.summary.fn += 1
        else:  # If's a True Positive or True Negative
            if data.target == data.prediction:
                self.summary.tp += 1
            else:
                self.summary.fp += 1

    def finish_epoch_summary(self):
        self.summary.model_name = self.name
        self.summary.epoch = self.epoch_curr_number

        self.epoch_curr_number += 1
        self.summary.total_samples = self.sample_count
        #self.summary.total_absolute_error= self.converge_tae_current
        self.iteration_num = 0  # Reset counter back to zero
        self.epoch_summaries.append(self.summary)
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



