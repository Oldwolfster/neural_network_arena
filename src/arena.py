import matplotlib.pyplot as plt
import random
import importlib
from tabulate import tabulate
from abc import ABC, abstractmethod

############################################################
# Arena Parameters are set here as global variables.       #
############################################################
epochs_to_run = 111     # Number of times training run will cycle through all training data
qty_rand_data = 30      # If random data is generated, how many
logs_to_print = 10      # It will print this many of the first and this many of the last iteration logs


def main():
    training_data = generate_random_linear_data(True)

    # List of tuples with module and class names
    nn_modules_and_classes = [
        'Simpletron'
        ,'SimpletronWithBias'
        #,'SimpletronWithReLU'
        #,'SimpletronWithExperiment'
        #,'SimpletronGradientDescent'
        #,'SimpletronWithL1L2Regularization'

    ]

    # Instantiate and train each NN
    metrics_list = []
    for nn_name in nn_modules_and_classes:

        # Define the folder name
        folder_name = 'gladiators'

        # Neural network name (assuming the class name is the same as the filename without .py)
        #nn_name = 'nn1'  # Example: 'nn1' corresponds to 'nn1.py' and the class 'nn1'

        # Full module name with folder path
        full_module_name = f'{folder_name}.{nn_name}'

        # Import the module
        print (f'Importing {full_module_name}')
        module = importlib.import_module(full_module_name)

        nn_class = getattr(module, nn_name)
        metrics = Metrics(nn_name)  # Create a new Metrics instance with the name as a string
        metrics_list.append(metrics)
        nn_instance = nn_class(epochs_to_run, metrics)
        nn_instance.train(training_data)
        give_me_a_line()
        print_results(metrics_list, training_data)
        # Precision is correctly predicted positive observations




def generate_random_linear_data(include_anomalies):
    training_data = []
    for _ in range(qty_rand_data):
        score = random.randint(1, 100)
        if include_anomalies:
            second_number = 1 if random.random() < (score / 100) else 0
        else:
            second_number = 1 if score >=.5 else 0
        training_data.append((score, second_number))
    return training_data


def give_me_a_line():
    print(f"{'=' * 129}")

def print_results(metrics_list, training_data):
    print_logs(metrics_list)
    print_grid(metrics_list)
    print(training_data)
    #print_footer(training_data, metrics)


def print_grid(metrics_list):
    # Prepare headers
    headers = ["Neural Network Arena", "Correct", "Wrong", "Loss", "Accuracy", "Precision", "Recall", "F1 Score"]

    # Prepare data
    data = []
    for metrics in metrics_list:
        # Calculate metrics
        accuracy = metrics.accuracy * 100
        precision = metrics.precision * 100
        recall = metrics.recall * 100
        f1_score = metrics.f1_score * 100

        # Append row data
        data.append([
            metrics.name,
            metrics.correct,
            metrics.wrong,
            f"{metrics.total_loss_for_epoch:.0f}",
            f"{accuracy:.2f}%",
            f"{precision:.2f}%",
            f"{recall:.2f}%",
            f"{f1_score:.2f}%"
        ])

    # Print table
    print(tabulate(data, headers=headers, tablefmt="grid"))


def print_logs(metrics_list):
    for metrics in metrics_list:
        # Print logs for each metrics object
        print(f"Logs for {metrics.name}:")
        for log in metrics.log_list[:logs_to_print]:
            print(log)
            give_me_a_line()

        for log in metrics.log_list[-logs_to_print:]:
            print(log)
            give_me_a_line()

def print_footer(training_data, metrics):
    print(training_data)

    calculate_anomaly_rate(training_data, metrics)
    anomalies = metrics.anomalies
    anom_percent = anomalies / metrics.data_count * 100

    print(f"# Anomalies:\t{anomalies:.0f} out of {metrics.data_count:.0f}\t {anom_percent}%")
    error_percent = 100-metrics.percents[-1]
    if anomalies > 0:
        error_rate_per_anomaly = error_percent / anomalies
        print(f"Error_rate per Anomaly:\t{error_rate_per_anomaly:.0f}%) # \tAnomalies:{anomalies}\tError Rate Per Anomaly:{error_rate_per_anomaly:.2f}")
    else:
        print("No anomalies!!!")

    #if show_graph == 1:
    #    plot_loss_and_weight_stability(metrics)


def calculate_anomaly_rate(training_data, metrics):
    count = 0
    anoms = 0
    for i, (credit_score, result) in enumerate(training_data):
        count += 1
        if is_it_an_anomaly(credit_score, result):
            anoms += 1

    metrics.anomalies = anoms
    metrics.data_count = count


def is_it_an_anomaly(credit_score, result):
    if credit_score >= 50 and result == 0:
        return True
    return False


class Metrics:
    def __init__(self, name):
        self.name = name
        self.total_loss_for_epoch = 0       # sum of absolute value of the loss for all iterations in an epoch
        self.data_count = 0                 # quantity of training data in current epoch.
        self.anomalies = 0                  # Anomaly is defined as outcome that is not the statistically most likely.. At the moment we can only detect this when we generate random data.
        self.losses = []                    # List of total loss for each of epoch.
        self.percents = []                  # Overall accuracy of entire epoch as a percentage (101 is the goal!)
        self.weights_this_epoch = []        # For the current  epoch, list of weights at each iteration.
        self.weights_last_epoch = []        # For the previous epoch, list of weights at each iteration.
        self.log_list = []                  # List of multiline string with details of the iteration
        self.predictions = []               # List of the result the model predicted.
        self.actuals = []                   # List of the actual result in the training data

    def clear_epoch_level(self):
        self.total_loss_for_epoch = 0
        self.predictions = []
        self.actuals = []
        self.weights_this_epoch.clear()

    @property
    def tp(self):
        return sum(1 for p, a in zip(self.predictions, self.actuals) if p == 1 and a == 1)

    @property
    def tn(self):
        return sum(1 for p, a in zip(self.predictions, self.actuals) if p == 0 and a == 0)

    @property
    def fp(self):
        return sum(1 for p, a in zip(self.predictions, self.actuals) if p == 1 and a == 0)

    @property
    def fn(self):
        return sum(1 for p, a in zip(self.predictions, self.actuals) if p == 0 and a == 1)

    @property
    def correct(self):
        return self.tp + self.tn

    @property
    def wrong(self):
        return self.fp + self.fn

    @property
    def precision(self):
        if (self.tp + self.fp) > 0:
            return self.tp / (self.tp + self.fp)
        return 0

    @property
    def recall(self):
        if (self.tp + self.fn) > 0:
            return self.tp / (self.tp + self.fn)
        return 0

    @property
    def f1_score(self):
        precision = self.precision
        recall = self.recall
        if (precision + recall) > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0

    @property
    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        if total > 0:
            return (self.tp + self.tn) / total
        return 0

    def add_epoch_metrics(self, learning_rate, training_time):
        self.losses.append(self.total_loss_for_epoch)
        self.percents.append(self.accuracy * 100)
        self.weights_last_epoch = self.weights_this_epoch[:]
        self.weights_this_epoch.clear()
        # Add additional metrics to lists if needed

class Gladiator(ABC):

    def __init__(self, number_of_epochs):
        self.number_of_epochs = number_of_epochs

    @abstractmethod
    def train(self, training_data):
        pass



if __name__ == '__main__':
    main()