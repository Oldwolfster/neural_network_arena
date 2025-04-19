import math
import statistics
import time
from typing import Tuple
from src.engine.Utils import dynamic_instantiate, set_seed

from .SQL import retrieve_training_data
from .SQL import record_training_data

from src.ArenaSettings import *
from src.ArenaSettings import run_previous_training_data
from .StoreHistory import record_snapshot
from .TrainingData import TrainingData
from src.engine.Reporting import generate_reports
from src.engine.Reporting import prep_RamDB
from ..Legos.LossFunctions import *
from ..NeuroForge.NeuroForge import *
import os
import importlib


def discover_gladiators(package="src.gladiators.regression"):
    """Dynamically load all gladiators from a given package."""
    path = package.replace(".", "/")
    gladiator_files = [
        f[:-3] for f in os.listdir(path)
        if f.endswith(".py") and not f.startswith("__")
    ]
    gladiators = []
    for name in gladiator_files:
        mod = importlib.import_module(f"{package}.{name}")
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and issubclass(obj, Gladiator) and obj is not Gladiator:
                gladiators.append(obj)
    return gladiators

def run_all_matchups(training_pit, shared_hyper):
    all_gladiators = discover_gladiators()
    for i, A in enumerate(all_gladiators):
        for B in all_gladiators[i+1:]:
            print(f"\n⚔️  {A.__name__} vs {B.__name__}")
            run_batch_of_matches(
                gladiators=[A(), B()],
                training_pit=training_pit,
                shared_hyper=shared_hyper,
                number_of_matches=10
            )

def run_batch_of_matches(gladiators, training_pit, shared_hyper, number_of_matches):
    """
    Runs multiple training sessions (matches) to account for stochastic variability.
    This helps validate optimizer stability, convergence behavior, and final accuracy.

    Args:
        gladiators (list): List of Gladiator classes or factory functions to instantiate models.
        training_pit (Arena): The data arena used to generate training data.
        shared_hyper (Config): Shared hyperparameters like learning rate, loss function, etc.
        number_of_matches (int): How many independent training runs to execute.

    Notes:
        - Seeds can be randomized each time or incremented.
        - Results are automatically logged via SQL integration.
    """
    print(f"⚔️ Running {number_of_matches} matches for each gladiator...")
    for match_index in range(number_of_matches):
        print(f"  ➤ Match {match_index + 1}/{number_of_matches}")

        # Optional: Change the seed slightly each match if you want to test randomness
        shared_hyper.random_seed += 1  # Or random.randint(...) if you want total chaos

        run_a_match(gladiators, training_pit, shared_hyper)



def grid_search_learning_rate(gladiators, training_pit, base_hyper, lr_values):
    """
    Try each learning rate in `lr_values`, run a match,
    and return the best LR plus all results.
    """
    results = []
    for lr in lr_values:
        # make a fresh copy of the hyperparameters (so per‐run seeds, db, etc. are isolated)
        hyper = copy.deepcopy(base_hyper)
        hyper.learning_rate = lr

        print(f"\n=== Grid search: trying learning_rate={lr} ===")
        infos = _run_single_match(gladiators, training_pit, hyper)

        # suppose you choose model 0's convergence time as your metric:
        metric = sum(info.seconds for info in infos) / len(infos)
        results.append((lr, metric))

    # pick the lr with minimal average run‐time (or replace with your own metric)
    best_lr, best_metric = min(results, key=lambda x: x[1])
    print(f"\n*** Best learning_rate={best_lr} (avg seconds={best_metric}) ***")
    return best_lr, results


def run_a_match(gladiators, training_pit, shared_hyper):

    seed = set_seed(shared_hyper.random_seed)

    # Shared resources
    db = prep_RamDB(gladiators)
    training_data = get_training_data(shared_hyper)
    training_data.arena_name = training_pit
    record_training_data(training_data.get_list())

    print()
    model_configs = []
    model_info_list = []
    for gladiator in gladiators:
        set_seed(seed)

        print(f"Preparing to run model: {gladiator}")

        # Create a unique config per model
        model_config        = Config(
            hyper           = shared_hyper,
            db              = db,               # Shared database
            training_data   = training_data,    # Shared training data
            gladiator_name  = gladiator,
        )
        model_config.set_defaults()
        #model_config.training_data = training_data # WHY DID I NEED THIS


        # Instantiate and train the model
        nn = dynamic_instantiate(gladiator, 'coliseum\\gladiators', model_config)
        start_time = time.time()

        # Actually train model
        model_config.full_architecture, snapshot = nn.train()

        #Record training details
        model_config.architecture = model_config.full_architecture[1:] #Remove inputs, keep hidden (if any) and output
        model_config.seconds = time.time() - start_time
        model_details= ModelInfo(gladiator, model_config.seconds, model_config.cvg_condition, model_config.full_architecture, model_config.training_data.problem_type )
        model_info_list.append(model_details)
        model_config.db.add(model_details)    #Writes record to ModelInfo table
        snapshot.runtime_seconds = model_config.seconds
        record_snapshot(snapshot)
        # Store Config for this model
        model_configs.append(model_config)

        # Easy place for quick dirty sql
        #model_config.db.query_print("SELECT * FROM        WeightAdjustments where nid = 0 and weight_index = 0")
        print(f"{gladiator} completed in {model_config.seconds} based on:{model_config.cvg_condition}")

    # Generate reports and send all model configs to NeuroForge
    generate_reports(db, training_data, shared_hyper, model_info_list)
    print(f"🛠️ Using Random Seed: {seed}")

    if shared_hyper.run_neuroForge:
        neuroForge(model_configs)

def get_training_data(hyper):
    # Check if Arena Settings indicates to retrieve and use past training_data
    if len(run_previous_training_data) > 0:
        return retrieve_training_data(run_previous_training_data)
    # If still here, do a run with new training data

    # Instantiate the arena and retrieve data
    arena               = dynamic_instantiate(training_pit, 'coliseum\\arenas', hyper.training_set_size)
    arena.arena_name    = training_pit

    result              = arena.generate_training_data_with_or_without_labels()             # Place holder to do any needed analysis on training data
    labels              = []
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


