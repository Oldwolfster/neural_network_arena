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


def DELETEEMErun_a_match_orig(gladiators, training_pit):
    config                  = Config(
        hyper   = HyperParameters(),
        db      = prep_RamDB()   # Create a connection to an in-memory SQLite database
    )

    seed                    = set_seed(config.hyper.random_seed)
    config.training_data    =  get_training_data(config.hyper)
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
        nn = dynamic_instantiate(gladiator, 'gladiators', model_config)
        start_time = time.time()

        # Actually train model
        model_config.cvg_condition, model_config.full_architecture, snapshot = nn.train()

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
        #model_config.db.query_print("SELECT epoch, iteration from Weight group by epoch, iteration order by epoch, iteration")

        #recorded_frames = model_config.db.query("SELECT epoch, iteration from Weight group by epoch, iteration order by epoch, iteration",as_dict=False)
        #print("frames")
        #print( recorded_frames)

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
    arena               = dynamic_instantiate(training_pit, 'arenas', hyper.training_set_size)
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


def print_reproducibility_info():
    print("\n🧬 Reproducibility Snapshot")
    print("────────────────────────────")
    print(f"Arena:             {self.config.training_data.arena_name}")
    print(f"Gladiator:         {self.config.gladiator_name}")
    print(f"Architecture:      {self.config.architecture}")
    print(f"Problem Type:      {self.config.training_data.problem_type}")
    print(f"Loss Function:     {self.config.loss_function.__class__.__name__}")
    print(f"Hidden AF:         {self.config.hidden_activation.__name__}")
    print(f"Output AF:         {self.config.output_activation.__name__}")
    print(f"Weight Init:       {self.config.initializer.__name__}")
    print(f"Data Norm Scheme:  {self.config.training_data.normalization_scheme}")
    print(f"Seed:              {self.config.hyper.seed}")
    print(f"Learning Rate:     {self.config.hyper.learning_rate}")
    print(f"Epochs Run:        {self.config.db.get_epochs_ran(self.config.gladiator_name)}")
    print(f"Convergence Rule:  {self.config.cvg_condition}")
    print(f"Final Error:       {self.config.db.get_final_mae(self.config.gladiator_name):.4f}")
    print(f"Final Accuracy:    {self.config.db.get_final_accuracy(self.config.gladiator_name):.2%}")
    print(f"Runtime (secs):    {self.config.seconds:.2f}")

    # Optional hash to detect drift
    # from hashlib import sha1
    # weights_hash = sha1(str(self.get_final_weights()).encode()).hexdigest()[:8]
    # print(f"Weights Checksum:  {weights_hash}")

    print("────────────────────────────\n")
