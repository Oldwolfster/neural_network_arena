from dataclasses import dataclass


"""
A class to record and manage metrics for neural network training.

Usage:
1) Instantiate - does all prep needed
2) Each iteration call record_iteration_metrics()
3) At the end of each epoch call epoch_completed()
-- All metrics will be calculated and available
Option to improve performance when it gets slow.  use numpy array instead of list for predictions and actuals

I refactored my metrics class as i felt it was keeping duplicate information and just getting messy.
First, I'm open to any recommendations you have.  Below is some specific points i'd like to address...
1) I feel there is no point in calculating the MSE MAE right now because i have the information to do it at time of reporting using self.errors and self.squared_errors and self.iteration_count, so why store duplicate information
2)Is it possible to eliminate self.total_error and self.total_error_squared and populate self.errors and self.squared_errors use some fancy logic in record_epoch 
similar to return sum(1 for p, a in zip(self.predictions, self.actuals) if p == 0 and a == 1) 
3) What's a good method for checking for convergence
4) Anything else, no matter how nitpicky... 
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
    new_bias: float = 0


class Metrics:
    def __init__(self, name, conv_e, conv_t):

        # Run Level members
        self.name = name
        self.epoch_curr_number = 1          # Which epoch are we currently on.
        self.converge_epoch = 0             # Which epoch did it converge on.
        self.converge_required = conv_e     # How many epochs of no change are required before we call it convg
        self.converge_progress = 0          # Used to track how many epochs loss remains unchanged.
        self.converge_threshold = conv_t    # How close does the MAE have to be between epochs to consider it converged
        self.converge_mae = 0               # This is the prior loss we are checking for stability against and if unchanged for n epochs then we call it converged.
        self.converge_mae_final=0           # The MAE the training session converged at.
        self.run_time = 0                   # How long did training take?
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

    def record_iteration(self, data):       # TODO Add APE AND PAP
        if self.epoch_is_over == 1:
            self.epoch_reset(data.epoch)

        self.iteration_count += 1
        self.predictions.append(data.prediction)
        self.targets.append(data.target)       # print(f"Appending Result {result} and Prediction {prediction}")

        # OLd logic for text log log_entry = f"{data.epoch }:{data.iteration }\tInput: {data.input:,.2f}\tTarget: {data.target:,.2f}\tPrediction: {data.prediction:,.2f}\t{self.pap(data):,.0f}\terror: {self.error(data):.2f}\tOld Weight: {data.weight:,.2f}\tNew Weight: {data.new_weight:,.2f}"
        # OLd logic for text log self.log_list.append(log_entry)
        self.log_data.append([f"{data.epoch}",
                         f"{data.iteration}",
                         f"{data.input:,.2f}",
                         f"{data.target:,.2f}",
                         f"{data.prediction:,.2f}",
                         f"{self.pap(data):,.0f}",
                         f"{self.error(data):.2f}",
                         f"{data.weight:,.2f}",
                         f"{data.new_weight:,.2f}",
                         f"{data.bias:,.4f}",
                         f"{data.new_bias:,.4f}"])

        #self.log_data.append( [data.epoch, data.iteration,data.input, data.target,data.prediction, self.error(data),data.weight,data.new_weight ])
        self.weights_this_epoch.append(data.new_weight)
        self.bias_values_this_epoch.append(data.new_bias)
        self.total_pap += self.pap(data)

    def epoch_reset(self, epoch_number):
        """When a new epoch begins, reset values"""
        self.epoch_is_over = 0          # Reset flag
        self.iteration_count = 0
        self.total_pap = 0
        self.weights_this_epoch.clear()
        self.bias_values_this_epoch.clear()
        self.epoch_curr_number = epoch_number
        self.predictions.clear()
        self.targets.clear()


    def record_epoch(self):
        ##### Does not apply to regression -- self.percents.append(self.accuracy * 100)
        self.epoch_is_over = 1  # signal to clear if another epoch begins (but retain last epoch values if it doesn't

        # Calculate and store total absolute error and squared error for this epoch
        total_absolute_error = sum(abs(p - t) for p, t in zip(self.predictions, self.targets))
        total_squared_error = sum((p - t) ** 2 for p, t in zip(self.predictions, self.targets))

        # Append calculated errors to respective lists
        self.errors.append(total_absolute_error)
        self.errors_squared.append(total_squared_error)
        # print(f"weights this epoch{self.weights_this_epoch}")
        cnt = self.iteration_count
        summary = [
            self.epoch_curr_number,  # Current epoch number
            self.iteration_count,  # Total number of iterations in this epoch
            self.total_pap,  # Total PAP for the epoch
            self.mape(),
            self.r_squared(),
            total_absolute_error,  # Total absolute error (TAE) for the epoch
            total_absolute_error / self.iteration_count,  # Mean absolute error (MAE)
            total_squared_error / self.iteration_count,  # Mean squared error (MSE)
            self.weights_this_epoch[-1],  # List of weights in this epoch
            self.bias_values_this_epoch[-1],  # data.bias if len(self.weights_this_epoch) > 0 else None,  # Bias at the end of epoch (if applicable)
        ]

        # Append this summary list to the epoch_summaries
        self.epoch_summaries.append(summary)

        mae = total_absolute_error / self.iteration_count
        return self.check_for_epoch_convergence(mae)

    def check_for_epoch_convergence(self, mae):
        if self.converge_epoch != 0:     # Prevent from running after convergence has been detected
            # Can i raise a warning here?
            return False

        if mae == 0:
            mae=.0000001                      # Prevent divide by zero errors

        difference_between_epochs = abs(self.converge_mae - mae)/mae
        if difference_between_epochs < self.converge_threshold: #Epoch MAE is within threshold of prior epoch
            self.converge_progress += 1
            if self.converge_progress > self.converge_required:
                self.converge_epoch = self.epoch_curr_number - self.converge_required - 1
                self.converge_mae_final = mae
                self.remove_logs_after_convergence()
                return True
        else:
            self.converge_progress = 0
            self.converge_mae = mae
        return False

    def remove_logs_after_convergence(self):
        print(f"***Converged listlen ={len(self.epoch_summaries)} self.converge_required={self.converge_required}")
        epochs_to_remove = self.converge_required + 1
        iterations_to_remove = epochs_to_remove * self.iteration_count
        del self.epoch_summaries[-epochs_to_remove:]
        del self.log_data[-iterations_to_remove:]


    def mape(self):
        return sum(abs((t - p) / t) for t, p in zip(self.targets, self.predictions) if t != 0) / len(self.targets) * 100

    def r_squared(self):
        y_mean = sum(self.targets) / len(self.targets)
        ss_tot = sum((y - y_mean) ** 2 for y in self.targets)
        ss_res = sum((y - p) ** 2 for y, p in zip(self.targets, self.predictions))
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def error(data):
        return abs(data.prediction - data.target)


    @staticmethod
    def pap(data):
        """
        PAP is a custom metric, prediction accuracy percentage
        It's used as an iteration to gauge the accuracy of a prediction
        For example, if the prediction is 99 OR 101 and the target is 100, PAP = 99 because it is off by 1%
        """
        if data.target == 0:
            data.target = .000001
        return (data.target - abs(data.target - data.prediction)) / data.target * 100



