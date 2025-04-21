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
from src.ArenaSettings import *

class NeuroEngine:
    def __init__(self):
        self.shared_hyper       = HyperParameters()
        self.seed               = set_seed(self.shared_hyper.random_seed)
        self.training_data      = None
        self.arena              = None
        self.db                 = None

        self.initialize_turbo()

    def initialize_turbo(self):
        # Shared resources
        set_seed(self.seed)     #TODO consider renaming reset_seed
        self.db = prep_RamDB(gladiators)
        self.training_data = self.instantiate_arena()

    def grid_search_learning_rate(self):
        return 0.0069


    def atomic_train_a_model(self, gladiator):
        set_seed(self.seed)
        model_config        = Config(hyper= self.shared_hyper,db = self.db,   training_data   = self.training_data,    gladiator_name  = gladiator,)
        model_config.set_defaults(self.training_data)
        if model_config.learning_rate == 0:
            model_config.learning_rate = self.grid_search_learning_rate()

        print(f"ABOUT TO TRAIN: gladiator {model_config.gladiator_name} LR is {model_config.learning_rate}")
        start_time = time.time()
        nn = dynamic_instantiate(gladiator, 'coliseum\\gladiators', model_config)

        # Actually train model
        last_mae = nn.train()       #Most info is stored in config

        #Record training details
        model_config.seconds = time.time() - start_time
        model_info = ModelInfo(gladiator, model_config.seconds, model_config.cvg_condition, model_config.full_architecture, model_config.training_data.problem_type )
        record_snapshot(model_config, last_mae)        # Store Config for this model
        model_config.db.add(model_info)              #Writes record to ModelInfo table
        print(f"🛠️ Using Random Seed: {self.seed}")
        return model_info, model_config

    def run_a_match(self, gladiators):
        model_configs       = []
        model_infos         = []
        for gladiator in    gladiators:
            info, config    = self.atomic_train_a_model(gladiator)
            model_infos     . append(info)
            model_configs   . append(config)

            # Easy place for quick dirty sql
            #model_config.db.query_print("SELECT * FROM        WeightAdjustments where nid = 0 and weight_index = 0")
            print(f"{gladiator} completed in {config} based on:{config.cvg_condition}")

        # Generate reports and send all model configs to NeuroForge
        generate_reports(self.db, self.training_data, self.shared_hyper, model_infos)

        if self.shared_hyper.run_neuroForge:
            neuroForge(model_configs)

    def instantiate_arena(self):
        # Check if Arena Settings indicates to retrieve and use past training_data
        if len(run_previous_training_data) > 0:
            return retrieve_training_data(run_previous_training_data)
        # If still here, do a run with new training data

        # Instantiate the arena and retrieve data
        arena               = dynamic_instantiate(training_pit, 'coliseum\\arenas', self.shared_hyper.training_set_size)
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

        # Assign the labels to hyperparameters and return
        self.shared_hyper.data_labels = labels
        td.arena_name = training_pit
        record_training_data(td.get_list())
        return td


