import os
import pprint
import time
from src.engine.Utils import dynamic_instantiate, set_seed
from .SQL import record_training_data
from .StoreHistory import record_results
from .TrainingBatchInfo import TrainingBatchInfo
from .TrainingData import TrainingData
from src.engine.Reporting import generate_reports, create_weight_tables
from src.engine.Reporting import prep_RamDB
from .TrainingRunInfo import TrainingRunInfo
from ..NeuroForge.NeuroForge import *
from src.ArenaSettings import *
from src.ArenaSettings import *



class NeuroEngine:   # Note: one different standard than PEP8... we align code vertically for better readability and asthetics
    def __init__(self, hyper):
        self.db                     = prep_RamDB()
        self.shared_hyper           = hyper
        self.seed                   = set_seed(self.shared_hyper.random_seed)
        self.training_data          = None
        self.test_attribute         = None #TODO DELETE ME
        self.test_strategy          = None #TODO DELETE ME

    def run_a_batch(self):          # lr_sweeps = self.check_for_learning_rate_sweeps(gladiators)   # Eventually Move this to TrainingBatchInfo
        TRIs: List[TrainingRunInfo] = []
        batch                       = TrainingBatchInfo(gladiators,arenas, dimensions)
        #print(f"batch={batch}")
        for i, setup in enumerate(batch.setups):
            print(f"Training Model with these settings: {setup}")
            if not setup.get("lr_specified", False): #feels backwards but is correct
                setup["learning_rate"] = self.learning_rate_sweep(setup)
            TRIs.append(self.atomic_train_a_model(setup, 3, epochs=0, run_id=i))                 # pprint.pprint(batch.ATAMs)
            print(f"MAE of {TRIs[-1].get("lowest_mae")}")
        generate_reports            (self.db, TRIs[0].training_data)
        TRIs[0].db.list_tables(3)
        neuroForge                  (TRIs[-4:])


    def atomic_train_a_model(self, setup, record_level: int, epochs=0, run_id=0): #ATAM is short for  -->atomic_train_a_model
            set_seed                    ( self.seed)
            training_data               = self.instantiate_arena(setup["arena"]) # Passed to base_gladiator through TRI
            set_seed                    ( self.seed)    #Reset seed as it likely was used in training_data
            create_weight_tables        ( self.db, setup["gladiator"])

            TRI                         = TrainingRunInfo(self.shared_hyper, self.db, training_data, setup, self.seed, run_id)
            NN                          = dynamic_instantiate(setup["gladiator"], 'coliseum\\gladiators', TRI)
            NN.train(epochs)            # Actually train model
            record_results              ( TRI, record_level)        # Store Config for this model #TODO make use of RecordLevel
            return                      TRI

    def learning_rate_sweep(self, setup: dict) -> float:
        """
        Sweep learning rates from 1.0 down to 1e-12 (logarithmically).
        Stops early if no improvement after `patience` trials.
        Modifies only 'learning_rate' in the setup dict.
        """
        gladiator    = setup.get("gladiator")
        start_lr     = 1e-6
        min_lr       = 1e-14
        max_lr       = 1
        factor       = 10
        max_trials   = 20

        best_error   = float("inf")
        best_lr      = None

        lr           = start_lr


        while lr >= min_lr and lr < max_lr:
            setup["learning_rate"] = lr
            TRI = self.atomic_train_a_model(setup,  record_level=0, epochs=20)
            error = TRI.get("total_error_for_epoch")
            if error > 1e20 and factor == 10:        #Try in the middle - if gradient explosion
                factor = .01                       # reverse direction
            print(f"😈Gladiator: {gladiator} - LR: {lr:.1e} → Last MAE: {error:.5f}")
            if error < best_error:
                best_error = error
                best_lr = lr
            lr *= factor
        print(f"\n🏆 Best learning_rate = {best_lr:.1e} (last_mae = {best_error:.5f})")
        return best_lr


    def instantiate_arena(self, arena):
        # Instantiate the arena and retrieve data
        arena               = dynamic_instantiate(arena, 'coliseum\\arenas', self.shared_hyper.training_set_size)
        arena.arena_name    = arena
        src                 = arena.source_code
        result              = arena.generate_training_data_with_or_without_labels()             # Place holder to do any needed analysis on training data
        labels              = []
        if isinstance(result, tuple):
            data, labels = result
            td = TrainingData(data, labels)  # Handle if training data has labels
            td.source_code = src
        else:
            data = result
            td = TrainingData(data)  # Handle if training data does not have labels
            td.source_code = src

            # Create default labels based on the length of a sample tuple
            sample_length = len(data[0]) if data else 0
            labels = [f"Input #{i + 1}" for i in range(sample_length - 1)]  # For inputs
            if sample_length > 0:
                labels.append("Target")  # For the target

        # Assign the labels to hyperparameters and return
        self.shared_hyper.data_labels = labels
        td.arena_name = training_pit
        #Deprecated - use random seed instead record_training_data(td.get_list())
        return td