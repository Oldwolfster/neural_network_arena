import os
import sys
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import importlib
import math
import statistics
from typing import Tuple

from engine.SQL import retrieve_training_data
from engine.SQL import record_training_data
from Reporting import print_results, determine_problem_type
import numpy as np
import time
from ArenaSettings import *


def run_a_match(gladiators, training_pit):
    hyper           = HyperParameters()
    mgr_list        = []
    raw_trn_data    = get_training_data(hyper)
    training_data   = TrainingData(to_list = raw_trn_data)             # Place holder to do any needed analysis on training data
    record_training_data(training_data.to_list)
    for gladiator in gladiators:    # Loop through the NNs competing.
        print(f"Preparing to run model:{gladiator}")
        nn = dynamic_instantiate(gladiator, 'Gladiators', gladiator, hyper)

        start_time = time.time()  # Start timing
        metrics_mgr = nn.train(training_data.to_list)
        mgr_list.append(metrics_mgr)
        end_time = time.time()  # End timing
        metrics_mgr.run_time = end_time - start_time
        print (f"{gladiator} completed in {metrics_mgr.run_time}")

    print_results(mgr_list, training_data.to_list, hyper, training_pit)


def get_training_data( hyper):
    # Check if Arena Settings indicates to retrieve and use past training_data
    if len(run_previous_training_data) > 0:
        return retrieve_training_data(run_previous_training_data)
        #return [(3.0829800228956428, 4.48830093538644, 30.780635057213185), (19.394768240791976, 4.132484554096511, 99.9506658661515)]
    # If still here, do a run with new training data
    arena = dynamic_instantiate(training_pit, 'Arenas', hyper.training_set_size)
    return arena.generate_training_data()

def dynamic_instantiate(class_name, path, *args):
    """
    Dynamically instantiate object without needing an include statement

    Args:
        class_name (str): The name of the file AND class to instantiate.
                            THE NAMES MUST MATCH EXACTLY
        path (str): The path to the module containing the class.
        *args: Arguments to pass to the class constructor.

    Returns:
        object: An instance of the specified class.
    """
    module_path = f'{path}.{class_name}'
    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)
    return class_(*args)


def generate_linearly_separable_data_ClaudeThinksWillLikeGradientDescent(n_samples=1000):
    # Generate two clusters of points
    cluster1 = np.random.randn(n_samples // 2, 1) - 2
    cluster2 = np.random.randn(n_samples // 2, 1) + 2

    X = np.vstack((cluster1, cluster2)).flatten()
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]

    return list(zip(X, y))

def calculate_loss_gradient(self, error: float, input: float) -> float:
    """
    Compute the gradient based on the selected loss function (MSE, MAE, RMSE, Cross Entropy, Huber).
    """
    if self.loss_function == 'MSE':
        # Mean Squared Error: Gradient is error * input
        return error * input
    elif self.loss_function == 'RMSE':
        # Root Mean Squared Error has the same gradient as MSE for individual updates
        return error * input
    elif self.loss_function == 'MAE':
        # Mean Absolute Error: Gradient is sign of the error * input
        return (1 if error >= 0 else -1) * input
    elif self.loss_function == 'Cross Entropy':
        # Convert raw prediction to probability using sigmoid
        pred_prob = 1 / (1 + math.exp(-((input * self.weight) + self.bias)))
        # Calculate binary cross-entropy gradient
        return (pred_prob - input) * input  # Gradient for cross-entropy
    elif self.loss_function == 'Huber':
        # Huber Loss: behaves like MSE for small errors and MAE for large errors
        delta = 1.0  # You can adjust this threshold depending on your dataset
        if abs(error) <= delta:
            # If error is small, use squared loss (MSE-like)
            return error * input
        else:
            # If error is large, use absolute loss (MAE-like)
            return delta * (1 if error > 0 else -1) * input
    else:
        # Default to MSE if no valid loss function is provided
        return error * input

@dataclass
class TrainingData: #Todo this class is not functioning properly.  it should be classifying each sample as outlier or not
    to_list: Tuple[float, ...]  # Multiple inputs
    #target: float
    is_outlier: bool = False

def identify_outlier(td: TrainingData):
    # Extract targets (last element of each tuple)
    print("Greetings!!!!!!!!!!!!")
    targets = [sample[-1] for sample in td.data]

    # Calculate mean and standard deviation of targets
    mean_target = statistics.mean(targets)
    stdev_target = statistics.stdev(targets)

    # Check for each sample if the target is beyond 3 standard deviations from the mean
    for sample in td.data:
        target_value = sample[-1]
        if abs(target_value - mean_target) > 3 * stdev_target:
            td.is_outlier = True
            break  # If any target is an outlier, mark the data as an outlier
        else:
            td.is_outlier = False
