import sqlite3
from json import dumps
from typing import List

from .RamDB import RamDB
from .Utils import IterationResult
from src.ArenaSettings import HyperParameters
from .TrainingData import TrainingData
from .Utils_DataClasses import Iteration
from src.engine.convergence.ConvergenceDetector import ConvergenceDetector


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
        self.db.add(iteration_data)
        self.abs_error_for_epoch += abs(iteration_data.error)

        for neuron in self.neurons:
            epoch_num=iteration_data.epoch
            iteration_num=iteration_data.iteration
            self.db.add(neuron, model=self.model_id, epoch_n = epoch_num, iteration_n = iteration_num )

    def finish_epoch(self):
        mae = self.abs_error_for_epoch / self.training_data.sample_count
        self.abs_error_for_epoch = 0 # Reset for next epoch
        print(f"MgrSQL ===> MAE = {mae}")
        return self.converge_detector.check_convergence(mae)



