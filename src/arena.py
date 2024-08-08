import random
import importlib
from metrics import *
from reporting import print_results
import numpy as np
import time

############################################################
# Arena Parameters are set here as global variables.       #
############################################################
epochs_to_run = 100     # Number of times training run will cycle through all training data
qty_rand_data = 3000      # If random data is generated, how many
display_graphs = False   # Display Graphs at the end of run
############################################################
# Model Parameters are set here as global variables.       #
############################################################
# Ideally avoid overriding these, but specific models, may need so must be free to do so
# It keeps comparisons straight if respected
default_neuron_weight   = .2        # Any value works as the training data will adjust it
default_learning_rate   = .1       # Reduces impact of adjustment to avoid overshooting
############################################################
# Interesting Data.       #
############################################################
# MAE 12, drops to 8 then coverges at 12 Training Data: [(74, 1), (67, 0), (56, 1), (22, 1), (24, 0), (64, 1), (18, 0), (48, 1), (72, 1), (79, 1), (48, 0), (86, 1), (78, 1), (44, 0), (9, 0), (78, 0), (92, 1), (72, 1), (29, 1), (15, 0), (56, 0), (88, 0), (85, 0), (96, 1), (96, 1), (45, 1), (7, 0), (96, 1), (59, 1), (86, 1)]


def main():
    training_data = generate_random_linear_data(True)
    # training_data = generate_linearly_separable_data_ClaudeThinksWillLikeGradientDescent()

    # For right now, set the gladiators here.
    gladiators = [

        '_Template_Simpletron'
        # , 'Simpletron_Fool'
        , 'Simpletron_LearningRate001'
        # ,'Simpletron_Bias'
        # ,'Simpletron_Gradient_Descent_Claude'
        # ,'SimpletronWithReLU'
        # ,'SimpletronWithExperiment'
        # ,'SimpletronGradientDescent'
        # ,'SimpletronWithL1L2Regularization'
    ]

    for x in range(30):
        run_a_match(gladiators, training_data)


def run_a_match(gladiators, arena):
    metrics_list = []
    for gladiator in gladiators:    # Loop through the NNs competing.
        metrics = Metrics(gladiator)  # Create a new Metrics instance with the name as a string
        metrics_list.append(metrics)
        nn = cut_a_cookie(gladiator, 'gladiators', epochs_to_run, metrics, default_neuron_weight, default_learning_rate)
        start_time = time.time()  # Start timing
        nn.train(arena)
        end_time = time.time()  # End timing
        metrics.run_time = end_time - start_time

    print_results(metrics_list, arena, display_graphs)

def run_a_match2(gladiators, arena):
    metrics_list = []
    for gladiator in gladiators:

        print(f'Importing gladiators.{gladiator}')
        module = importlib.import_module(f'gladiators.{gladiator}')
        nn_class = getattr(module, gladiator)
        metrics = Metrics(gladiator)  # Create a new Metrics instance with the name as a string
        metrics_list.append(metrics)
        nn_instance = nn_class(epochs_to_run, metrics)

        start_time = time.time()  # Start timing
        nn_instance.train(arena)
        end_time = time.time() # End timing
        metrics.run_time = end_time - start_time

    print_results(metrics_list, arena, display_graphs)


def cut_a_cookie(class_name, path, *args):
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

def generate_random_linear_data(include_anomalies):
    training_data = []
    for _ in range(qty_rand_data):
        score = random.randint(1, 100)
        if include_anomalies:
            second_number = 1 if random.random() < (score / 100) else 0
        else:
            second_number = 1 if score >=50 else 0
        training_data.append((score, second_number))
    return training_data


if __name__ == '__main__':
    main()