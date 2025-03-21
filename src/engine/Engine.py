import math
import statistics
import time
from typing import Tuple
from src.engine.Utils import dynamic_instantiate, set_seed

from .SQL import retrieve_training_data
from .SQL import record_training_data

from src.ArenaSettings import *
from src.ArenaSettings import run_previous_training_data
from .TrainingData import TrainingData
from src.engine.Reporting import generate_reports
from src.engine.Reporting import prep_RamDB
from ..Legos.LossFunctions import *
from ..NeuroForge.NeuroForge import *




def run_a_match_orig(gladiators, training_pit):

    config = ModelConfig(
        hyper = HyperParameters(),
        db =    prep_RamDB()   # Create a connection to an in-memory SQLite database
    )

    seed            = set_seed(config.hyper.random_seed)
    config.training_data =  get_training_data(config.hyper)
    record_training_data(config.training_data.get_list())

    print()
    model_info_list = [] # Initialize an empty list to store ModelInfo objects #TODO remove me
    model_configs = []
    for gladiator in gladiators:    # Loop through the NNs competing.
        set_seed(seed)      #reset for each gladiator
        print(f"Preparing to run model:{gladiator}")
        config.gladiator_name = gladiator
        #nn = dynamic_instantiate(gladiator, 'gladiators', gladiator, hyper, training_data, db)
        nn = dynamic_instantiate(gladiator, 'gladiators',config)

        start_time = time.time()  # Start timing
        cvg_condition,full_architecture = nn.train()
        end_time = time.time()  # End timing
        run_time = end_time - start_time
        model_details= ModelInfo(gladiator, run_time, cvg_condition, full_architecture, config.training_data.problem_type )
        config.db.add(model_details)    #TODO this looks wrong
        model_info_list.append(model_details)

        print (f"{gladiator} completed in {run_time}")

    generate_reports(config.db, config.training_data, config.hyper, model_info_list)
    print(f"🛠️ Using Random Seed: {seed}")
    if config.hyper.run_neuroForge:
        #neuroForge(config.db, config.training_data, config.hyper, model_info_list)
        neuroForge(config, model_info_list)



def run_a_match(gladiators, training_pit):
    shared_hyper = HyperParameters()  # Create ONE instance
    seed = set_seed( shared_hyper.random_seed)

    # Shared resources
    db = prep_RamDB()
    training_data = get_training_data(shared_hyper)
    record_training_data(training_data.get_list())

    print()
    model_configs = []
    model_info_list = []
    for gladiator in gladiators:
        set_seed(seed)

        print(f"Preparing to run model: {gladiator}")

        # Create a unique config per model
        model_config = ModelConfig(
            hyper=shared_hyper,
            db=db,  # Shared database
            #training_data=training_data,  # Shared training data
            gladiator_name=gladiator
        )
        model_config.training_data = training_data


        # Instantiate and train the model
        nn = dynamic_instantiate(gladiator, 'gladiators', model_config)
        start_time = time.time()
        model_config.cvg_condition, model_config.full_architecture = nn.train()
        model_config.architecture = model_config.full_architecture[1:] #Remove inputs, keep hidden (if any) and output
        print(f"Gladiator:{gladiator}  model_config.full_architecture[1:]={model_config.full_architecture}\t model_config.architecture={model_config.architecture}")

        #TODO remove next 4 lines- currently needed  to create ModelInfo table.
        end_time = time.time()  # End timing
        run_time = end_time - start_time
        model_details= ModelInfo(gladiator, run_time, model_config.cvg_condition, model_config.full_architecture, model_config.training_data.problem_type )

        model_info_list.append(model_details)
        model_config.db.add(model_details)    #Writes record to ModelInfo table

        model_config.seconds = time.time() - start_time

        print(f"{gladiator} completed in {model_config.seconds}")

        # Store ModelConfig for this model
        model_configs.append(model_config)

    # Generate reports and send all model configs to NeuroForge
    generate_reports(db, training_data, shared_hyper, model_info_list)
    print(f"🛠️ Using Random Seed: {seed}")

    if shared_hyper.run_neuroForge:
        neuroForge(model_configs)



def get_training_data(hyper):
    # Check if Arena Settings indicates to retrieve and use past training_data
    if len(run_previous_training_data) > 0:
        return retrieve_training_data(run_previous_training_data)
        #return [(3.0829800228956428, 4.48830093538644, 30.780635057213185), (19.394768240791976, 4.132484554096511, 99.9506658661515)]
    # If still here, do a run with new training data

    #return TrainingData(arena.generate_training_data_with_or_without_labels())             # Place holder to do any needed analysis on training data
    # Instantiate the arena and retrieve data
    arena = dynamic_instantiate(training_pit, 'arenas', hyper.training_set_size)
    #result = TrainingData(arena.generate_training_data_with_or_without_labels())             # Place holder to do any needed analysis on training data
    result = arena.generate_training_data_with_or_without_labels()             # Place holder to do any needed analysis on training data
    #print(f"result={result}")
    labels = []
    if isinstance(result, tuple):
        data, labels = result
        td = TrainingData(data)  # Handle if training data has labels
    else:
        data = result
        td = TrainingData(data)  # Handle if training data does not have labels

        # Create default labels based on the length of a sample tuple
        sample_length = len(data[0]) if data else 0
        labels = [f"Input #{i + 1}" for i in range(sample_length - 1)]  # For inputs
        if sample_length > 0:
            labels.append("Target")  # For the target

    # Assign the labels to hyperparameters
    hyper.data_labels = labels
    return td

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
