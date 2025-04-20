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



def grid_search_learning_rate(model_config):
    """
    Sweep learning rates and pick the best based on last_mae.
    Starts low and increases logarithmically.
    """
    results = []
    start_lr = 1e-6
    stop_lr = 10
    factor = 10

    model_config.is_exploratory = True

    try:
        lr = start_lr
        while lr < stop_lr:
            model_config.learning_rate = lr
            try:
                nn = dynamic_instantiate(model_config.gladiator_name, 'coliseum\\gladiators', model_config)
                last_mae = nn.train(10, lr)
                print(f"🔎 Tried learning_rate={lr:.1e}, last_mae={last_mae:.4f}")
            except OverflowError as e:
                print(f"💥 OverflowError at lr={lr:.1e}: {e}")
                last_mae = float('inf')  # Treat exploded LRs as worst possible
            except Exception as e:
                print(f"⚠️  Unexpected error at lr={lr:.1e}: {e}")
                last_mae = float('inf')  # Same fallback

            results.append((lr, last_mae))
            lr *= factor

        print("\n📋 Learning Rate Sweep Results:")
        for lr, mae in results:
            print(f"  - LR: {lr:.1e} → Last MAE: {mae:.5f}")

        best_lr, best_metric = min(results, key=lambda x: x[1])
        print(f"\n🏆 Best learning_rate={best_lr:.1e} (last_mae={best_metric:.4f})")
        model_config.learning_rate = best_lr

    finally:
        model_config.is_exploratory = False





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

        # Instantiate and train the model
        nn = dynamic_instantiate(gladiator, 'coliseum\\gladiators', model_config)
        print(f"for gladiator {model_config.gladiator_name} LR is {model_config.default_lr}")
        if model_config.default_lr == 0.0:  #Learning rate is not set, do a sweep
            grid_search_learning_rate(model_config)

        # Actually train model
        start_time = time.time()
        last_mae = nn.train()       #Most info is stored in config
        print(f"last mae from real match{last_mae}")
        #Record training details
        model_config.seconds = time.time() - start_time
        model_details= ModelInfo(gladiator, model_config.seconds, model_config.cvg_condition, model_config.full_architecture, model_config.training_data.problem_type )
        model_info_list.append(model_details)
        model_config.db.add(model_details)              #Writes record to ModelInfo table
        record_snapshot( model_config, last_mae)        # Store Config for this model
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


