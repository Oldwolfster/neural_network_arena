from src.engine.Metrics import Metrics
from Utils import EpochSummary
from typing import List



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
        self.epoch_curr_number = 1          # Which epoch are we currently on.
        Metrics.set_acc_threshold(acc_t)    # Set at Class level (not instance) one value shared across all instances
        self.converge_epoch = 0             # Which epoch did it converge on.
        self.converge_required = conv_e     # How many epochs of no change are required before we call it convg
        self.converge_progress = 0          # Used to track how many epochs loss remains unchanged.
        self.converge_threshold = conv_t    # How close does the MAE have to be between epochs to consider it converged
        self.converge_prior_tae = 0         # This is the prior loss we are checking for stability against and if unchanged for n epochs then we call it converged.
        self.converge_tae_final=0           # The MAE the training session converged at.
        self.converge_tae_current = 0       # Sum the Total Absolute Error as we log each iteration
        self.metrics = []                   # The list of metrics this manager is running.

    def record_iteration(self, result):
        self.iteration_num += 1
        data = Metrics(result)
        self.metrics.append(data)
        self.update_summary(data)
        self.check_for_epoch_over(data)

    def update_summary(self, data: Metrics):
        self.summary.total_absolute_error += abs(data.prediction - data.target)
        self.summary.total_squared_error += (data.prediction - data.target) ^ 2

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


    def check_for_epoch_over(self, data):
        self.converge_tae_current += abs(data.target - data.prediction)
        if data.iteration == self.sample_count:
            self.finish_summary(data)
            self.check_for_convergence()


    def finish_summary(self, data : Metrics):
        self.summary.model_name = self.name
        self.summary.epoch = data.epoch
        self.summary.total_samples = self.iteration_num
        self.summary.total_absolute_error= self.converge_tae_current
        self.iteration_num = 0  # Reset counter back to zero
        self.epoch_summaries.append(self.summary)
        self.summary = EpochSummary()

    def check_for_convergence(self):
        if self.converge_epoch != 0:  # Prevent from running after convergence has been detected
            return False
        tae = self.converge_prior_tae
        if tae == 0:
            tae = 1E-20  # Prevent divide by zero errors

        difference_between_epochs = abs(self.converge_prior_tae - tae) / tae
        if difference_between_epochs < self.converge_threshold:  # Epoch MAE is within threshold of prior epoch
            self.converge_progress += 1
            if self.converge_progress > self.converge_required:
                self.converge_epoch = self.epoch_curr_number - self.converge_required - 1
                self.converge_tae_final = tae
                self.remove_logs_after_convergence()
                return True
        else:
            self.converge_progress = 0
            self.converge_prior_tae = tae
        return False

    def remove_logs_after_convergence(self):
        return
        epochs_to_remove = self.converge_required + 1
        iterations_to_remove = epochs_to_remove * self.iteration_count
        print(f"***iteration_count ={self.iteration_count}\tepoch_curr_number={self.epoch_curr_number}\tepoch_curr_number={iterations_to_remove}")
        #del self.epoch_summaries[-epochs_to_remove:]
        #del self.log_data[-iterations_to_remove:]
        #del self.metrics[-iterations_to_remove:]

    def iteration_log(self) -> List[List]:
        return [metric.to_list() for metric in self.metrics]
    """
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
        """

