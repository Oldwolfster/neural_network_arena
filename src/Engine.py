import random
import importlib
from MetricsMgr import MetricsMgr
from Reporting import print_results, determine_problem_type
import numpy as np
import time
from ArenaSettings import *




def main():

    run_a_match(gladiators, training_pit)


def run_a_match(gladiators, training_pit):
    mgr_list = []
    arena_data = dynamic_instantiate(training_pit, 'Arenas', training_set_size)
    training_data = arena_data.generate_training_data()
    #problem_type = determine_problem_type(training_data)
    #print(f"In Arena {problem_type}")
    for gladiator in gladiators:    # Loop through the NNs competing.
        metrics_mgr = MetricsMgr(gladiator, training_set_size, converge_epochs, converge_threshold, accuracy_threshold, arena_data)  # Create a new Metrics instance with the name as a string
        mgr_list.append(metrics_mgr)
        nn = dynamic_instantiate(gladiator, 'Gladiators', epochs_to_run, metrics_mgr, default_neuron_weight, default_learning_rate)
        start_time = time.time()  # Start timing
        nn.train(training_data)
        end_time = time.time()  # End timing
        metrics_mgr.run_time = end_time - start_time
        print (f"{gladiator} completed in {metrics_mgr.run_time}")

    print_results(mgr_list, training_data, display_graphs, display_logs, display_train_data ,display_epoch_sum, epochs_to_run, training_set_size, default_learning_rate, training_pit)
    #print_results(mgr_list, training_data, display_graphs, display_logs, display_train_data ,display_epoch_sum, epochs_to_run, training_set_size)


# , 'Simpletron_LearningRate001'
# ,'Simpletron_Bias'
# ,'Simpletron_Gradient_Descent_Claude'
# ,'SimpletronWithReLU'
# ,'SimpletronWithExperiment'
# ,'SimpletronGradientDescent'
# ,'SimpletronWithL1L2Regularization'

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



if __name__ == '__main__':
    main()