import os
from src.engine.Utils import smart_format
import pprint
import time
from enum import Enum
import psutil
from src.engine.Utils import dynamic_instantiate, set_seed
from .SQL import record_training_data
from .StoreHistory import record_results
from .TrainingBatchInfo import TrainingBatchInfo
from .TrainingData import TrainingData
from src.engine.Reporting import generate_reports, create_weight_tables
from src.engine.Reporting import prep_RamDB
from .TrainingRunInfo import TrainingRunInfo
from .Utils_DataClasses import RecordLevel
from ..NeuroForge.NeuroForge import *
from src.ArenaSettings import *
from src.ArenaSettings import *
"""
Part of this refactor is revamping the sql
Tracking that here.
1) DONE: Model_Info
2) Neuron
3) Iteration
4) Weight Adj tbl 1 WeightAdjustments_update_
5) Weight Adj Tbl 2 WeightAdjustments_finalize_
6) ErrorSignalCalcs
7) Weight
ModelInfo
"""



class NeuroEngine:   # Note: one different standard than PEP8... we align code vertically for better readability and asthetics
    def __init__(self, hyper):
        self.db                     = prep_RamDB()
        self.shared_hyper           = hyper
        self.seed                   = set_seed(self.shared_hyper.random_seed)
        self.training_data          = None
        #self.test_attribute         = None #TODO DELETE ME
        #self.test_strategy          = None #TODO DELETE ME

    def run_a_batch(self):
        TRIs: List[TrainingRunInfo] = []
        batch                       = TrainingBatchInfo(gladiators, arenas, dimensions)
        while True:
            setup                   = batch.mark_done_and_get_next_config()
            if setup is None:         break
            print                   ( f"\n💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪💪")
            print                   ( f"💪 {batch.id_of_current} of {batch.id_of_last}-{setup["arena"]} - these settings: {setup}")
            if not setup.get        ( "lr_specified", False):   setup["learning_rate"] = self.learning_rate_sweep(setup)
            record_level            = RecordLevel.FULL if batch.id_of_current <= self.shared_hyper.nf_count else RecordLevel.SUMMARY
            TRI                     = self.atomic_train_a_model(setup, record_level, epochs=0, run_id=batch.id_of_current)
            if record_level         ==RecordLevel.FULL: TRIs.append(TRI)

        if TRIs:
            print("🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬")
            print("🔬🔬 Loading Neuroforge... 🔬🔬")
            print("🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬🔬")
            TRIs[0].db.copy_tables_to_permanent()
            if self.shared_hyper.display_train_data: print(f"Training Data: {TRIs[0].training_data.raw_data}")
            #neuroForge(TRIs)

    def atomic_train_a_model(self, setup, record_level: RecordLevel, epochs=0, run_id=0): #ATAM is short for  -->atomic_train_a_model
            set_seed                    ( self.seed)
            training_data               = self.instantiate_arena(setup["arena"]) # Passed to base_gladiator through TRI
            set_seed                    ( self.seed)    #Reset seed as it likely was used in training_data
            create_weight_tables        ( self.db, run_id)
            TRI                         = TrainingRunInfo(self.db, training_data, self.shared_hyper.training_set_size,self.shared_hyper.epochs_to_run, setup, self.seed, run_id, record_level)
            NN                          = dynamic_instantiate(setup["gladiator"], 'coliseum\\gladiators', TRI)
            NN.train(epochs)            # Actually train model
            record_results              (TRI)        # Store Config for this model #TODO make use of RecordLevel
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

        # ─── NEW: early‐stop patience setup ───────────────────────────
        patience          = 3     # number of consecutive no‐improve steps before quitting
        no_improve_count  = 0
        # ───────────────────────────────────────────────────────────────

        best_error   = float("inf")
        best_lr      = None
        lr           = start_lr
        trials       = 0
        print(f"\t😈😈 Welcome to the Learning Rate Sweep.  Heads up, Below info is 'LR:→MAE' repeated😈😈"),
        while lr >= min_lr and lr < max_lr and trials < max_trials:
            setup["learning_rate"] = lr
            TRI = self.atomic_train_a_model(setup, RecordLevel.NONE, epochs=20)
            error = TRI.mae
            #print(f"\tLR:{lr:.1e} → {smart_format(error)}", end="")
            print(f"\t😈\tLR:{lr:.1e} → {smart_format(error)}")

            # ─── check for gradient explosion ───────────────────────────
            if (error is None or error > 1e20) and factor == 10:           # gradient explosion – no need to search higher
                factor = 0.1                            # reverse direction
                lr      = 1e-6

                # Reset patience once we flip direction
                no_improve_count = 0
                continue
            # ─────────────────────────────────────────────────────────────

            # ─── track best_error and reset or increment patience ──────
            if error is not None and error < best_error:
                best_error      = error
                best_lr         = lr
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience and best_lr is not None:
                #print(f"⌛ No improvement in {patience} consecutive trials—stopping early.")
                break
            # ─────────────────────────────────────────────────────────────

            lr *= factor
            trials += 1

        print(f"\t😈\t🏆🏆🏆 Best learning_rate = {best_lr:.1e} (last_mae = {best_error:.5f} 🏆🏆🏆)\n")
        return best_lr

    def learning_rate_sweep2(self, setup: dict) -> float:
        """
        Sweep learning rates log-scale from start_lr down to min_lr
        and from start_lr up to max_lr if needed. Early-stop via patience.
        """
        gladiator    = setup.get("gladiator")
        start_lr     = 1e-6
        min_lr       = 1e-14
        max_lr       = 1.0
        factor_up    = 10
        factor_down  = 0.1
        patience     = 3
        max_trials   = 20

        best_error      = float("inf")
        best_lr         = None
        no_improve      = 0
        trials          = 0

        # Stage 1: sweep up from start_lr → max_lr
        lr = start_lr
        factor = factor_up
        while trials < max_trials and lr <= max_lr:
            setup["learning_rate"] = lr
            TRI = self.atomic_train_a_model(setup, record_level=0, epochs=20)
            error = TRI.mae

            # Treat any missing/inf MAE as “explosion”
            if error is None or not math.isfinite(error) or error > 1e10:
                # Flip direction: go downward from start point
                factor = factor_down
                lr     = start_lr   # reset to start and begin downward sweep
                no_improve = 0
                continue

            print(f"😈 {gladiator} – LR: {lr:.1e} → MAE: {error:.5f}")

            # Bookkeeping best/patience
            if error < best_error:
                best_error = error
                best_lr    = lr
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⌛ No improvement in {patience} trials—stopping.")
                    break

            trials += 1
            lr *= factor

        # Stage 2 (optional): if we never tested below start_lr in stage 1, do a down sweep
        if (best_lr is None or best_lr >= start_lr) and trials < max_trials:
            # Only sweep downward if we haven’t found anything better below start_lr
            lr = start_lr * factor_down
            factor = factor_down
            while trials < max_trials and lr >= min_lr:
                setup["learning_rate"] = lr
                TRI = self.atomic_train_a_model(setup, record_level=0, epochs=20)
                error = TRI.mae

                # If we hit explosion again (unlikely at tiny rates), break
                if error is None or not math.isfinite(error) or error > 1e10:
                    break

                print(f"😈 {gladiator} – LR: {lr:.1e} → MAE: {error:.5f}")

                if error < best_error:
                    best_error = error
                    best_lr    = lr
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"⌛ No improvement in {patience} trials—stopping downward sweep.")
                        break

                trials += 1
                lr *= factor

        # Finally, if nothing ever worked, fallback to start_lr
        if best_lr is None:
            best_lr = start_lr
            print(f"⚠️ No valid learning rate found; defaulting to {start_lr:.1e}")

        print(f"\n🏆 Best learning_rate = {best_lr:.1e} (MAE = {best_error:.5f})")
        return best_lr

    def learning_rate_sweepMay31(self, setup: dict) -> float:
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
            error = TRI.mae


            print(f"😈Gladiator: {gladiator} - LR: {lr:.1e} → Last MAE: {error:.5f}")
            if error > 1e20 and factor == 10:           #gradient explosion - no need to search higher
                factor = .1                       # reverse direction
                lr      = 1e-6

            if error < best_error:
                best_error = error
                best_lr = lr
            lr *= factor
        print(f"\n🏆 Best learning_rate = {best_lr:.1e} (last_mae = {best_error:.5f})")
        return best_lr


    def instantiate_arena(self, arena_name):
        # Instantiate the arena and retrieve data
        arena               = dynamic_instantiate(arena_name, 'coliseum\\arenas', self.shared_hyper.training_set_size)
        arena.arena_name    = arena_name
        src                 = arena.source_code
        #print(f"src={src}")
        result              = arena.generate_training_data_with_or_without_labels()             # Place holder to do any needed analysis on training data
        feature_labels              = []
        if isinstance(result, tuple):
            #data, feature_labels = result
            #td = TrainingData(data, feature_labels)  # Handle if training data has labels
            #td.source_code = src
            data = result[0]
            feature_labels = result[1] if len(result) > 1 else []
            target_labels = result[2] if len(result) > 2 else []
            td = TrainingData(data, feature_labels, target_labels)  # Handle if training data has labels
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
        #self.shared_hyper.data_labels = labels
        td.arena_name = arena_name
        #Deprecated - use random seed instead record_training_data(td.get_list())
        return td