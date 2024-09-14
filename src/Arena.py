import random
import importlib
from Metrics import *
from Reporting import print_results
import numpy as np
import time

############################################################
# Battle Parameters are set here as global variables.       #
############################################################
epochs_to_run       = 5            # Number of times training run will cycle through all training data
training_set_size   = 5           # Qty of training data
default_neuron_weight   = .2        # Any initial value works as the training data will adjust it
default_learning_rate   = .001      # Affects magnitude of weight adjustments

############################################################
# Report Parameters are set here as global variables.      #
############################################################
display_graphs      = False         # Display Graphs at the end of run
display_logs        = True          # Display the logs at the end of the run
display_train_data  = True          # Display the training data at the end of the rn.


def main():

    # Set the training pit here
    training_pit = "QuadraticInput_CreditScore"
    #training_pit = "SingleInput_CreditScore"
    training_pit = "QuadraticTrainingPit"
    training_pit = "HouseValue_SqrFt"
    # List the gladiators here
    gladiators = [
        '_Template_Simpletron'
        ,'Simpletron_Sigmoid'

    ]

    run_a_match(gladiators, training_pit)


def run_a_match(gladiators, training_pit):
    metrics_list = []
    arena_data = dynamic_instantiate(training_pit, 'TrainingPits', training_set_size)
    training_data = arena_data.generate_training_data()

    for gladiator in gladiators:    # Loop through the NNs competing.
        metrics = Metrics(gladiator)  # Create a new Metrics instance with the name as a string
        metrics_list.append(metrics)
        nn = dynamic_instantiate(gladiator, 'Gladiators', epochs_to_run, metrics, default_neuron_weight, default_learning_rate)
        start_time = time.time()  # Start timing
        nn.train(training_data)
        end_time = time.time()  # End timing
        metrics.run_time = end_time - start_time
        print (f"{gladiator} completed in {metrics.run_time}")

    print_results(metrics_list, training_data, display_graphs, display_logs, display_train_data, epochs_to_run, training_set_size)


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




if __name__ == '__main__':
    main()