import random
import importlib
from metrics import *
from reporting import print_results

############################################################
# Arena Parameters are set here as global variables.       #
############################################################
epochs_to_run = 100     # Number of times training run will cycle through all training data
qty_rand_data = 30000      # If random data is generated, how many
display_graphs = False   # Display Graphs at the end of run

############################################################
# Interesting Data.       #
############################################################
# MAE 12, drops to 8 then coverges at 12 Training Data: [(74, 1), (67, 0), (56, 1), (22, 1), (24, 0), (64, 1), (18, 0), (48, 1), (72, 1), (79, 1), (48, 0), (86, 1), (78, 1), (44, 0), (9, 0), (78, 0), (92, 1), (72, 1), (29, 1), (15, 0), (56, 0), (88, 0), (85, 0), (96, 1), (96, 1), (45, 1), (7, 0), (96, 1), (59, 1), (86, 1)]


def main():
    for x in range(30):
        run_a_match()
def run_a_match():
    training_data = generate_random_linear_data(True)

    # List of tuples with module and class names
    gladiators = [

        #'Simpletron_Fool'
        #'Simpletron_LearningRate001'
        ,'Simpletron'
        ,'Simpletron_Bias'
        ,'Simpletron_Bias_Claude'
        #,'SimpletronWithReLU'
        #,'SimpletronWithExperiment'
        #,'SimpletronGradientDescent'
        #,'SimpletronWithL1L2Regularization'
    ]

    # Instantiate and train each NN
    metrics_list = []
    for nn_name in gladiators:

        # Define the folder name
        folder_name = 'gladiators'

        full_module_name = f'{folder_name}.{nn_name}' # Full module name with folder path

        # Import the module
        print (f'Importing {full_module_name}')
        module = importlib.import_module(full_module_name)

        nn_class = getattr(module, nn_name)
        metrics = Metrics(nn_name)  # Create a new Metrics instance with the name as a string
        metrics_list.append(metrics)
        nn_instance = nn_class(epochs_to_run, metrics)
        nn_instance.train(training_data)

    print_results(metrics_list, training_data, display_graphs)





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