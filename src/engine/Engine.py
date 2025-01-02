import math
import statistics
import numpy as np
import time
from typing import Tuple

from .RamDB import RamDB
from .SQL import retrieve_training_data
from .SQL import record_training_data

from src.ArenaSettings import *
from src.engine.BaseArena import BaseArena
from src.engine.BaseGladiator import Gladiator
from src.ArenaSettings import run_previous_training_data
from .TrainingData import TrainingData
from src.engine.Reporting import generate_reports
from src.engine.Reporting import prep_RamDB
from .Utils_DataClasses import ModelInfo


def run_a_match(gladiators, training_pit):
    hyper           = HyperParameters()
    training_data   =  get_training_data(hyper)
    db =    prep_RamDB()   # Create a connection to an in-memory SQLite database
    record_training_data(training_data.get_list())

    for gladiator in gladiators:    # Loop through the NNs competing.
        print(f"Preparing to run model:{gladiator}")
        nn = dynamic_instantiate(gladiator, 'gladiators', gladiator, hyper, training_data, db)

        start_time = time.time()  # Start timing
        cvg_condtion = nn.train()
        end_time = time.time()  # End timing
        run_time = end_time - start_time
        model_details= ModelInfo(gladiator, run_time)
        db.add(model_details)
        print (f"{gladiator} completed in {run_time}")

    generate_reports(db, training_data, hyper)




def get_training_data(hyper):
    # Check if Arena Settings indicates to retrieve and use past training_data
    if len(run_previous_training_data) > 0:
        return retrieve_training_data(run_previous_training_data)
        #return [(3.0829800228956428, 4.48830093538644, 30.780635057213185), (19.394768240791976, 4.132484554096511, 99.9506658661515)]
    # If still here, d  o a run with new training data
    arena = dynamic_instantiate(training_pit, 'arenas', hyper.training_set_size)
    return TrainingData(arena.generate_training_data())             # Place holder to do any needed analysis on training data


import os
import importlib
import inspect

def dynamic_instantiate(class_name, base_path='arenas', *args):
    """
    Dynamically instantiate an object of any class inheriting from BaseArena
    or BaseGladiator in the specified file, avoiding class name mismatches.

    Args:
        class_name (str): The name of the file to search within (file must end in .py).
        base_path (str): The base module path to search within.
        *args: Arguments to pass to the class constructor.

    Returns:
        object: An instance of the specified class.

    Raises:
        ImportError: If the file or class is not found.
        ValueError: If the same file is found in multiple subdirectories or no matching class found.
    """
    # Set up the directory to search
    search_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), base_path.replace('.', os.sep))
    matched_module = None

    # Walk through directories to find the file
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            if file == f"{class_name}.py":
                # Calculate relative path from src folder and clean it up
                relative_path = os.path.relpath(root, os.path.dirname(os.path.dirname(__file__)))
                # Clean up extra ".." and slashes for importlib
                relative_path = relative_path.strip(os.sep).replace(os.sep, '.')
                module_path = f"{relative_path}.{class_name}"

                # Debugging output to verify paths
                print(f"Found file: {file}")
                print(f"Module path: {module_path}")

                # Check for duplicates
                if matched_module:
                    raise ValueError(f"Duplicate module found for {class_name}: {matched_module} and {module_path}")

                # Set matched module path
                matched_module = module_path

    if not matched_module:
        raise ImportError(f"Module {class_name} not found in {base_path} or any subdirectories.")

    # Import module and instantiate class
    module = importlib.import_module(matched_module)
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (issubclass(obj, BaseArena) or issubclass(obj, Gladiator)) and obj.__module__ == module.__name__:
            return obj(*args)

    raise ImportError(f"No class inheriting from BaseArena or BaseGladiator found in {class_name}.py")



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
class TrainingDataOld: #Todo this class is not functioning properly.  it should be classifying each sample as outlier or not
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
