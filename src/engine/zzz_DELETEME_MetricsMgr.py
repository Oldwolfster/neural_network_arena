from .Metrics import Metrics
from .Utils import EpochSummary, IterationResult
from src.ArenaSettings import HyperParameters
from .TrainingData import TrainingData


class MetricsMgr:       #(gladiator, training_set_size, converge_epochs, converge_threshold, accuracy_threshold, arena_data)  # Create a new Metrics instance with the name as a string
    def __init__       (self, name, hyper: HyperParameters, training_data: TrainingData):
        #def __init__(self, name, sample_count: int,  conv_e, conv_t, acc_t, data):
        # Run Level members
        self.training_data = training_data
        self.name = name
        self.hyper = hyper
        self.epoch_summaries = []           # Store the epoch summaries
        self.summary = EpochSummary()       # The current Epoch summary
        self.run_time = 0                   # How long did training take?
        self.iteration_num = 0              # Current Iteration #
        self.sample_count = 0               # Number of samples in each iteration.
        self.accuracy_threshold = (
            hyper.accuracy_threshold)     # In regression, how close must it be to be considered "accurate"
        Metrics.set_acc_threshold(
            hyper.accuracy_threshold)    # Set at Class level (not instance) one value shared across all instances
        self.epoch_curr_number = 1          # Which epoch are we currently on.
        #TODO name should indicate for current epoch
        self.metrics = []                   # The list of metrics this manager is running.
        #self.converge_detector              = ConvergenceDetector(hyper.converge_threshold, hyper.converge_epochs) #, training_data.sum_targets))
        self._converge_detector = None
        self.convergence_signal = []      # Will be set by convergence detector

        #Below added when switching to multiple neurons
        self.metrics_iteration = []
        self.metrics_neuron = []


    @property
    def converge_detector(self):
        """
        Provides lazy instantiation of converge_detector so it can pass it(CD) a copy of itself (MMgr)
        """
        if self._converge_detector is None:
            # Lazy import to avoid circular reference
            from src.engine.convergence.ConvergenceDetector import ConvergenceDetector
            self._converge_detector = ConvergenceDetector(self.hyper, self)
        return self._converge_detector

    def record_iteration(self, result : IterationResult):
        self.iteration_num += 1
        # I believe these two lines can be removed as it is now set in BaseGladiator
        if self.sample_count < self.iteration_num:
            self.sample_count = self.iteration_num

        data = Metrics(result)
        self.metrics.append(data)
        self.update_summary(data)


    def update_summary(self, data: Metrics):
        self.summary.total_error += data.prediction - data.target
        self.summary.total_absolute_error += abs(data.prediction - data.target)
        self.summary.total_squared_error += (data.prediction - data.target) ** 2
        self.summary.final_weight = data.new_weight
        self.summary.final_bias = data.new_bias
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

    def finish_epoch_summary(self):
        self.summary.model_name = self.name
        self.summary.epoch = self.epoch_curr_number

        self.epoch_curr_number += 1
        self.summary.total_samples = self.sample_count
        #self.summary.total_absolute_error= self.converge_tae_current
        self.iteration_num = 0  # Reset counter back to zero
        self.epoch_summaries.append(self.summary)
        #print(f"epoch:{self.summary.epoch}self.summary.total_absolute_error = {self.summary.total_absolute_error}")
        mae= self.summary.total_absolute_error / self.summary.total_samples
        #print (f"mean_absolute_error: {mae}")
        #epochs_to_remove = self.converge_detector.check_convergence(mae) #, self.summary.final_weight)

        if not self.converge_detector.check_convergence():
            self.summary = EpochSummary()   # Create summary for next epoch
            return False
        # Still here so it has converged
        return True

    def remove_converged_epochs_from_logs(self, epochs_to_remove):
        #print(f"epochs to remove:{epochs_to_remove}\tepochs :{epochs}")
        iterations_to_remove = epochs_to_remove * self.sample_count
        del self.epoch_summaries[-epochs_to_remove:]
        del self.metrics[-iterations_to_remove:]



    def finish_epoch_summaryDELETEME(self):
        self.summary.model_name = self.name
        self.summary.epoch = self.epoch_curr_number

        self.epoch_curr_number += 1
        self.summary.total_samples = self.sample_count
        #self.summary.total_absolute_error= self.converge_tae_current
        self.iteration_num = 0  # Reset counter back to zero
        self.epoch_summaries.append(self.summary)
        #print(f"epoch:{self.summary.epoch}self.summary.total_absolute_error = {self.summary.total_absolute_error}")
        mae= self.summary.total_absolute_error / self.summary.total_samples
        #print (f"mean_absolute_error: {mae}")
        epochs_to_remove = self.converge_detector.check_convergence(mae) #, self.summary.final_weight)
        if epochs_to_remove == 0:      # 0 indicates it has NOT converged
            self.summary = EpochSummary()   # Create summary for next epoch
            return False
        # Still here so it has converged
        self.remove_converged_epochs_from_logs(epochs_to_remove)
        return True
