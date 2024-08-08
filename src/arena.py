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
# Interesting Data.       #
############################################################
# MAE 12, drops to 8 then coverges at 12 Training Data: [(74, 1), (67, 0), (56, 1), (22, 1), (24, 0), (64, 1), (18, 0), (48, 1), (72, 1), (79, 1), (48, 0), (86, 1), (78, 1), (44, 0), (9, 0), (78, 0), (92, 1), (72, 1), (29, 1), (15, 0), (56, 0), (88, 0), (85, 0), (96, 1), (96, 1), (45, 1), (7, 0), (96, 1), (59, 1), (86, 1)]


def main():
    training_data = generate_random_linear_data(True)
    # training_data = generate_linearly_separable_data_ClaudeThinksWillLikeGradientDescent()

    # For right now, set the gladiators here.
    gladiators = [

        '_Template_Simpletron'
        , 'Simpletron_Fool'
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
    for nn_name in gladiators:
        print(f'Importing gladiators.{nn_name}')
        module = importlib.import_module(f'gladiators.{nn_name}')
        nn_class = getattr(module, nn_name)
        metrics = Metrics(nn_name)  # Create a new Metrics instance with the name as a string
        metrics_list.append(metrics)
        nn_instance = nn_class(epochs_to_run, metrics)

        start_time = time.time()  # Start timing
        nn_instance.train(arena)
        end_time = time.time() # End timing
        metrics.run_time = end_time - start_time

    print_results(metrics_list, arena, display_graphs)


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