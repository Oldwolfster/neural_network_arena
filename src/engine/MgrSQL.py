import sqlite3
from json import dumps
from typing import List

from .RamDB import RamDB

from src.ArenaSettings import HyperParameters
from .TrainingData import TrainingData
from .Utils_DataClasses import Iteration
from src.engine.convergence.ConvergenceDetector import ConvergenceDetector
from typing import Dict

class MgrSQL:       #(gladiator, training_set_size, converge_epochs, converge_threshold, accuracy_threshold, arena_data)  # Create a new Metrics instance with the name as a string
    def __init__(self, model_id, hyper: HyperParameters, training_data: TrainingData, neurons: List, ramDb: RamDB):
        # Run Level members
        self.training_data          = training_data
        self.model_id               = model_id
        self.hyper                  = hyper
        self.neurons                = neurons
        self.db                     = ramDb
        self.iteration_num          = 0                         # Current Iteration #
        self.epoch_curr_number      = 1                         # Which epoch are we currently on.
        self.sample_count           = len(self.training_data.get_list())          # Calculate and store sample count= 0               # Number of samples in each iteration.
        self.accuracy_threshold     = (hyper.accuracy_threshold)    # In regression, how close must it be to be considered "accurate"
        self.converge_detector     = ConvergenceDetector(hyper,training_data)
        self.abs_error_for_epoch = 0
        self.convergence_signal = None      # Will be set by convergence detector
    """
    @property    
    def converge_detector(self):
    
        Provides lazy instantiation of converge_detector so it can pass it(CD) a copy of itself (MMgr)
        
        if self._converge_detector is None:
            # Lazy import to avoid circular reference
            from src.engine.convergence.ConvergenceDetector import ConvergenceDetector
            self._converge_detector = ConvergenceDetector(self.hyper, self.training_data, self)
        return self._converge_detector
    """
    def record_iteration(self, iteration_data: Iteration):
        #print("****************************RECORDING ITERATION 2")
        #print(f"iteration_data.error={iteration_data.error}\titeration_data.absolute_error={iteration_data.absolute_error}")

        self.db.add(iteration_data)
        #self.db.query_print("Select * from Iteration")
        #print(f"error: {iteration_data.error} (type: {type(iteration_data.error)})")
        #print(f"absolute_error: {iteration_data.absolute_error} (type: {type(iteration_data.absolute_error)})")
        #self.db.query_print("PRAGMA table_info(Iteration);")
        self.abs_error_for_epoch += abs(iteration_data.error)   #TODO This shuold be using RamDB

        for neuron in self.neurons:
            epoch_num=iteration_data.epoch
            iteration_num=iteration_data.iteration
            self.db.add(neuron, model=self.model_id, epoch_n = epoch_num, iteration_n = iteration_num )

    def finish_epoch(self):
        mae = self.abs_error_for_epoch / self.training_data.sample_count
        self.abs_error_for_epoch = 0 # Reset for next epoch
        epoch_metrics = self.get_metrics_from_ramdb()
        #print(f"MgrSQL ===> MAE = {mae} from dict {epoch_metrics['mean_absolute_error']}")

        return self.converge_detector.check_convergence(epoch_metrics)

    def get_metrics_from_ramdb(self) -> Dict[str, float]:
        """
        Fetch the latest epoch's metrics for the current model.

        Returns:
            Dict[str, float]: A dictionary containing the metrics, where all values are floats.
        """
        sql = """
            SELECT *
            FROM EpochSummary
            WHERE model_id = ?
            ORDER BY epoch DESC
            LIMIT 1;
        """
        # Pass the parameter correctly as a tuple
        result = self.db.query(sql, params=(self.model_id,), as_dict=True)

        if result:
            return result[0]  # Return the first row as a dictionary
        raise RuntimeError("No records found for the specified model_id")  # Raise error if no records are found






