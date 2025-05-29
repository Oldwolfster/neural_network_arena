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

    def run_a_batchOrig(self):          # lr_sweeps = self.check_for_learning_rate_sweeps(gladiators)   # Eventually Move this to TrainingBatchInfo
        TRIs : TrainingRunInfo      = []
        batch                       = TrainingBatchInfo(gladiators,arenas, dimensions)
        for setup in batch.setups:  TRIs.append(self.atomic_train_a_model(setup, 3))                 # pprint.pprint(batch.ATAMs)
        print(f"MAE of{TRIs[-1].    get("lowest_mae")}: {setup}")
        generate_reports            (self.db, self.training_data)
        neuroForge                  (TRIs)

    def run_a_batch(self):          # lr_sweeps = self.check_for_learning_rate_sweeps(gladiators)   # Eventually Move this to TrainingBatchInfo
        TRIs: List[TrainingRunInfo] = []
        batch                       = TrainingBatchInfo(gladiators,arenas, dimensions)
        print(f"batch={batch}")
        for setup in batch.setups:
            print(f"setup={setup}")
            if not setup.get("lr_specified", False): #feels backwards but is correct
                setup["learning_rate"] = self.learning_rate_sweep(setup)
            TRIs.append(self.atomic_train_a_model(setup, 3))                 # pprint.pprint(batch.ATAMs)
            print(f"MAE of{TRIs[-1].    get("lowest_mae")}: {setup}")
        generate_reports            (self.db, TRIs[0].training_data)
        neuroForge                  (TRIs)


    def atomic_train_a_model(self, setup, record_level: int, epochs=0): #ATAM is short for  -->atomic_train_a_model
            set_seed                    ( self.seed)
            training_data               = self.instantiate_arena(setup["arena"]) # Passed to base_gladiator through TRI
            set_seed                    ( self.seed)    #Reset seed as it likely was used in training_data
            create_weight_tables        ( self.db, setup["gladiator"])

            TRI                         = TrainingRunInfo(self.shared_hyper, self.db, training_data, setup, self.seed)
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


    def learning_rate_sweepOLD(self, gladiator) -> float:
        """
        Sweep learning rates and pick the best based on last_mae.
        Starts low and increases logarithmically.
        """
        start_lr                    = 1e-6
        stop_lr                     = 10
        original_factor             = 10                #in case it switches directions, only switch once.
        factor                      = original_factor
        lr                          = start_lr
        min_lr_limit                = 1e-15  # hard stop

        results = []
        while lr < stop_lr and lr >= min_lr_limit:
            TRI               = self.atomic_train_a_model(gladiator, lr, 20) #Pass learning rate being swept

            print                   (f"Gladiator: {gladiator}  - LR: {lr:.1e} → Last MAE: {TRI.config.lowest_error:.5f}")
            results.append          ((lr, TRI.config.lowest_error))

            # 🔁 If we're still using the original direction, and the lowest LR blew up...
            if factor == original_factor and lr == start_lr and TRI.config.lowest_error > 1e5:
                print(f"🛑 MAE {config.lowest_error:.2e} too high at LR {lr:.1e}, reversing sweep direction...")
                factor = 0.1  # 🔄 now sweeping downward
            lr                      *= factor
        best_lr, best_metric        = min(results, key=lambda x: x[1])
        print                       (f"\n🏆 Best learning_rate={best_lr:.1e} (last_mae={best_metric:.4f})")
        return                      best_lr





    def check_gladiator_for_learning_rate_sweep(self, gladiator):
        # Create temp instantiation of model to see if Learning rate is specified.
        temp_TRI                = TrainingRunInfo(self.shared_hyper,self.db,self.training_data, gladiator, self.seed)
        #create_weight_tables    (self.db, gladiator)

        temp_nn                 = dynamic_instantiate(gladiator, 'coliseum\\gladiators', temp_TRI)
        print (f"temp_TRI.config.learning_rate is {temp_TRI.config.learning_rate}")

        #return self.learning_rate_sweep(gladiator)
    #########################################
    # Old way is below
    ###########################################
    def run_a_match(self, gladiators, arena, test_attribute=None, test_strategy=None):
        self.training_data  = self.instantiate_arena(arena)
        model_configs       = []
        model_infos         = []
        results             = []
        TRIs                = []    #Store the training run infos
        self.test_attribute = test_attribute
        self.test_strategy  = test_strategy


        for gladiator in    gladiators:
            print           (f"Preparing to run model: {gladiator}")
            set_seed        (self.seed)
            TRI             = self.atomic_train_a_model(gladiator) #Don't pass LR as we don't know it yet
            TRIs            . append(TRI)

            print           (f"{gladiator} completed in {TRI.config.seconds} based on:{TRI.config.cvg_condition} with relative accuracy of {TRI.config.accuracy_percent}")
            print(self.db.get_add_timing())
            results.append(TRI.config.accuracy_percent)

        # Generate reports and send all model configs to NeuroForge
        print(f"🛠️  Random Seed:    {self.seed}")
        generate_reports(self.db, self.training_data, self.shared_hyper, model_infos, arena)
        if self.shared_hyper.record: neuroForge(TRIs)
        return results

#    def create_fresh_config(self, gladiator):
#        return Config(hyper=self.shared_hyper,db=self.db, training_data=self.training_data, gladiator_name=gladiator)


    def atomic_train_a_model_ORIGINAL (self, gladiator, learning_rate=None, epochs=0):
        record_results              = (epochs == 0)  #if epochs is specified it is LR Sweep, don't record and clean up
        if learning_rate is None:   learning_rate = self.check_for_learning_rate_sweep(gladiator)
        TRI                         = TrainingRunInfo(self.shared_hyper,self.db,self.training_data, gladiator, self.seed)
        create_weight_tables        (self.db, gladiator)
        TRI.config.learning_rate  = learning_rate                         # Either from sweep or config if sweep found it was set in config
        set_seed                    (self.seed)
        nn                          = dynamic_instantiate(gladiator, 'coliseum\\gladiators', TRI)

        # 🧠 Inject test strategy if provided (e.g., test loss function, activation, etc.)

        TRI.config                  . set_defaults( self.test_attribute, self.test_strategy)
        # Actually train model
        last_mae                    = nn.train(epochs)
        TRI.config                  .configure_popup_headers()# MUST OCCUR AFTER CONFIGURE MODEL SO THE OPTIMIZER IS SET
        TRI                         . record_finish_time()
        model_info                  = ModelInfo(gladiator, TRI.config .seconds, TRI.config .cvg_condition, TRI.config .architecture, TRI.config .training_data.problem_type )
        #Record training details    #print(f"architecture = {model_config.architecture}")
        if record_results:
            record_snapshot         (TRI.config , last_mae, self.seed)        # Store Config for this model
            TRI.db.add     (model_info)              #Writes record to ModelInfo table
        return                     TRI

    def check_for_learning_rate_sweep(self, gladiator):
        # Create temp instantiation of model to see if Learning rate is specified.
        #temp_config             = self.create_fresh_config(gladiator)
        temp_TRI                = TrainingRunInfo(self.shared_hyper,self.db,self.training_data, gladiator, self.seed)
        create_weight_tables    (self.db, gladiator)

        temp_nn                 = dynamic_instantiate(gladiator, 'coliseum\\gladiators', temp_TRI)
        temp_TRI.config         . set_defaults( self.test_attribute, self.test_strategy)
        #temp_config             . set_defaults( self.test_attribute, self.test_strategy)
        # If LR is manually set in model, skip sweep
        #print(f"temp_config.learning_rate={temp_config.learning_rate}")
        if temp_TRI.config.learning_rate != 0.0:     return temp_TRI.config.learning_rate

        print(f"🌀 Running LEARNING RATE SWEEP for {gladiator}")
        return self.learning_rate_sweep(gladiator)

    def learning_rate_sweepOrig(self, gladiator) -> float:
        """
        Sweep learning rates and pick the best based on last_mae.
        Starts low and increases logarithmically.
        """
        start_lr                    = 1e-6
        stop_lr                     = 10
        original_factor             = 10                #in case it switches directions, only switch once.
        factor                      = original_factor
        lr                          = start_lr
        min_lr_limit                = 1e-15  # hard stop

        results = []
        while lr < stop_lr and lr >= min_lr_limit:
            TRI               = self.atomic_train_a_model(gladiator, lr, 20) #Pass learning rate being swept

            print                   (f"Gladiator: {gladiator}  - LR: {lr:.1e} → Last MAE: {TRI.config.lowest_error:.5f}")
            results.append          ((lr, TRI.config.lowest_error))

            # 🔁 If we're still using the original direction, and the lowest LR blew up...
            if factor == original_factor and lr == start_lr and TRI.config.lowest_error > 1e5:
                print(f"🛑 MAE {TRI.config.lowest_error:.2e} too high at LR {lr:.1e}, reversing sweep direction...")
                factor = 0.1  # 🔄 now sweeping downward
            lr                      *= factor
        best_lr, best_metric        = min(results, key=lambda x: x[1])
        print                       (f"\n🏆 Best learning_rate={best_lr:.1e} (last_mae={best_metric:.4f})")
        return                      best_lr


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