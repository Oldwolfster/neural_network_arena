import time
from src.engine.Utils import dynamic_instantiate, set_seed
from .SQL import record_training_data
from .StoreHistory import record_snapshot
from .TrainingData import TrainingData
from src.engine.Reporting import generate_reports, create_weight_tables
from src.engine.Reporting import prep_RamDB
from ..NeuroForge.NeuroForge import *
from src.ArenaSettings import *

class NeuroEngine:   # Note: one different standard than PEP8... we align code vertically for better readability and asthetics

    print_rules_once_per_gladiator = False #this is not working yet
    def __init__(self):
        self.db = prep_RamDB()
        self.shared_hyper       = HyperParameters()
        self.seed               = set_seed(self.shared_hyper.random_seed)
        self.training_data      = None

    def run_a_match(self, gladiators, arena):
        self.training_data  = self.instantiate_arena(arena)
        model_configs       = []
        model_infos         = []

        for gladiator in    gladiators:
            #NeuroEngine.print_rules_once_per_gladiator = False   # referenced in Config to surpress spam
            print           (f"Preparing to run model: {gladiator}")
            set_seed        (self.seed)
            info, config    = self.atomic_train_a_model(gladiator) #Don't pass LR as we don't know it yet
            model_infos     . append(info)
            model_configs   . append(config)
            print           (f"{gladiator} completed in {config} based on:{config.cvg_condition}")
            print(self.db.get_add_timing())

        # Generate reports and send all model configs to NeuroForge
        print(f"🛠️  Random Seed:    {self.seed}")
        generate_reports(self.db, self.training_data, self.shared_hyper, model_infos)
        neuroForge(model_configs)

    def create_fresh_config(self, gladiator):
        return Config(hyper=self.shared_hyper,db=self.db, training_data=self.training_data, gladiator_name=gladiator)


    def atomic_train_a_model(self, gladiator, learning_rate=None, epochs=None):
        record_results = epochs is None  #if epochs is specified it is LR Sweep, don't record and clean up
        if learning_rate is None:
            learning_rate = self.check_for_learning_rate_sweep(gladiator)

        model_config                = self.create_fresh_config(gladiator)
        create_weight_tables        (self.db, gladiator)
        self.delete_records         (self.db, gladiator) # in case it had been run by LR sweep
        start_time                  = time.time()

        model_config.learning_rate  = learning_rate                         # Either from sweep or config if sweep found it was set in config
        set_seed                    (self.seed)
        nn                          = dynamic_instantiate(gladiator, 'coliseum\\gladiators', model_config)
        model_config                . set_defaults()

        # Actually train model
        last_mae                    = nn.train(0 if record_results else epochs)
        model_config                .configure_popup_headers()# MUST OCCUR AFTER CONFIGURE MODEL SO THE OPTIMIZER IS SET
        model_config.seconds        = time.time() - start_time
        model_info                  = ModelInfo(gladiator, model_config.seconds, model_config.cvg_condition, model_config.architecture, model_config.training_data.problem_type )
        #Record training details    #print(f"architecture = {model_config.architecture}")
        if record_results:
            record_snapshot         (model_config, last_mae, self.seed)        # Store Config for this model
            model_config.db.add     (model_info)              #Writes record to ModelInfo table
        return                      model_info, model_config

    def check_for_learning_rate_sweep(self, gladiator):
        # Create temp instantiation of model to see if Learning rate is specified.
        temp_config             = self.create_fresh_config(gladiator)
        create_weight_tables    (self.db, gladiator)
        self.delete_records     (self.db, gladiator) # in case it had been run by LR sweep
        temp_nn                 = dynamic_instantiate(gladiator, 'coliseum\\gladiators', temp_config)
        temp_config             . set_defaults()
        # If LR is manually set in model, skip sweep
        print(f"temp_config.learning_rate={temp_config.learning_rate}")
        if temp_config.learning_rate != 0.0:
            return temp_config.learning_rate

        print(f"🌀 Running LEARNING RATE SWEEP for {gladiator}")
        return self.learning_rate_sweep(gladiator)

    def learning_rate_sweep(self, gladiator) -> float:
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
            _, config               = self.atomic_train_a_model(gladiator, lr, 20) #Pass learning rate being swept
            print                   (f"  - LR: {lr:.1e} → Last MAE: {config.lowest_error:.5f}")
            results.append          ((lr, config.lowest_error))

            # 🔁 If we're still using the original direction, and the lowest LR blew up...
            if factor == original_factor and lr == start_lr and config.lowest_error > 1e5:
                print(f"🛑 MAE {config.lowest_error:.2e} too high at LR {lr:.1e}, reversing sweep direction...")
                factor = 0.1  # 🔄 now sweeping downward
            lr                      *= factor
        best_lr, best_metric        = min(results, key=lambda x: x[1])
        print                       (f"\n🏆 Best learning_rate={best_lr:.1e} (last_mae={best_metric:.4f})")
        return                      best_lr

    def delete_records(self, db, gladiator_name, possible_columns=None):
        """
        Deletes records across all tables where one of the possible columns matches the given gladiator_name.

        Args:
            db: Your database connection or wrapper.
            gladiator_name (str): The model ID or name to delete.
            possible_columns (list of str, optional): Columns to check, in order of preference.
        """
        if possible_columns is None:
            possible_columns = ['model_id', 'model', 'gladiator']

        # Delete tables that have model_id in name rather than waste a column
        table_name = f"WeightAdjustments_update_{gladiator_name}"
        db.execute(f"DELETE FROM {table_name}")
        table_name = f"WeightAdjustments_finalize_{gladiator_name}"
        db.execute(f"DELETE FROM {table_name}")

        # Get list of all table names
        tables = db.query("SELECT name FROM sqlite_master WHERE type='table';")

        for table_row in tables:
            #ez_debug(table_row=table_row)
            table_name = table_row['name']

            # Get column names for this table
            columns = db.query(f"PRAGMA table_info({table_name});", as_dict = False)
            column_names = [col[1] for col in columns]

            # Find first matching column
            matching_column = next((col for col in possible_columns if col in column_names), None)

            if matching_column:
                #print(f"🧹 Deleting from {table_name} where {matching_column} = '{gladiator_name}'")
                #db.execute(f"DELETE FROM {table_name} WHERE {matching_column} = ?", (gladiator_name,))
                db.execute(f"DELETE FROM {table_name} WHERE {matching_column} = '{gladiator_name}'")

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