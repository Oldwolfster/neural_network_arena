import json
import sqlite3
from json import dumps
from typing import List

import numpy as np

from .Neuron import Neuron
from .RamDB import RamDB

from src.ArenaSettings import HyperParameters
from .TrainingData import TrainingData
from .Utils_DataClasses import Iteration
from src.engine.convergence.ConvergenceDetector import ConvergenceDetector
from typing import Dict

class MgrSQL:       #(gladiator, training_set_size, converge_epochs, converge_threshold, accuracy_threshold, arena_data)  # Create a new Metrics instance with the name as a string
    def __init__(self, config, model_id, hyper: HyperParameters, training_data: TrainingData, neurons: List, ramDb: RamDB):
        # Run Level members
        self.training_data          = training_data
        self.model_id               = model_id
        self.hyper                  = hyper
        self.neurons                = neurons
        self.db                     = ramDb
        self.config                 = config
        self.iteration_num          = 0                         # Current Iteration #
        self.epoch_curr_number      = 1                         # Which epoch are we currently on.
        self.sample_count           = len(self.training_data.get_list())          # Calculate and store sample count= 0               # Number of samples in each iteration.
        self.accuracy_threshold     = (hyper.accuracy_threshold)    # In regression, how close must it be to be considered "accurate"
        self.converge_detector     = ConvergenceDetector(hyper,training_data, config)
        self.abs_error_for_epoch = 0
        self.convergence_signal = None      # Will be set by convergence detector

    def should_record_sample(self, epoch: int, sample_index: int) -> bool:
        # ✅ Always record the first sample of every epoch
        if sample_index == 1:
            return True

        strategy = self.hyper.record_sample_strategy

        # ✅ If no override for this epoch, do not record
        if epoch not in strategy:
            return False

        # ✅ Record if the sample is explicitly listed (or all are, via -1)
        samples = strategy[epoch]
        return -1 in samples or sample_index in samples

    def record_iteration(self, iteration_data: Iteration, layers: List[List[Neuron]]):
        """
        Add the current iteration data to the database
        """

        epoch_num = iteration_data.epoch
        iteration_num = iteration_data.iteration
        #print(f"Deciding {epoch_num}\t {iteration_num}")
        #if not self.should_record_sample(epoch_num, iteration_num):
        #    return
        self.db.add(iteration_data)
        self.abs_error_for_epoch += abs(iteration_data.error)

        # Iterate over layers and neurons
        for layer_index, layer in enumerate(layers):

            for neuron in layer:
                if layer_index == 0:  # First hidden layer (takes raw sample inputs)
                    raw_inputs = json.loads(iteration_data.inputs)  # Parse JSON string to list
                    neuron.neuron_inputs = np.array(raw_inputs, dtype=np.float64)
                    #print(f"storing neuron data First Hidden Layer (Layer 0). nid={neuron.nid}, inputs={neuron.neuron_inputs}")
                else:   # All subsequent layers - NOTE: Output is not considered a layer in respect to these neurons
                    previous_layer = layers[layer_index - 1]
                    neuron.neuron_inputs = np.array(
                        [prev.activation_value for prev in previous_layer], dtype=np.float64
                    )
                    #print(f"storing neuron data Hidden Layer {layer_index}. nid={neuron.nid}, inputs={neuron.neuron_inputs}")
                    #for prev in previous_layer:
                    #    print(f"prev.nid={prev.nid}\t{prev.activation_value}")
                # Add the neuron data to the database
                #TODO take out 'input_tensor' and delta from exclude keys
                self.db.add(neuron, exclude_keys={"activation", "learning_rate"}, model=self.model_id, epoch_n=epoch_num, iteration_n=iteration_num)
        Neuron.bulk_insert_weights(db = self.db, model_id = self.model_id, epoch=epoch_num, iteration=iteration_num )


    def finish_epoch(self, epoch: int):
        mae = self.abs_error_for_epoch / self.training_data.sample_count
        if mae < self.config.lowest_error:    # New lowest error
            self.config.lowest_error = mae
            self.config.lowest_error_epoch = epoch


        self.abs_error_for_epoch = 0 # Reset for next epoch
        epoch_metrics = self.get_metrics_from_ramdb(epoch)
        #print(f"MgrSQL ===> MAE = {mae} from dict {epoch_metrics['mean_absolute_error']}")
        self.epoch_curr_number+=1
        return self.converge_detector.check_convergence(self.epoch_curr_number, epoch_metrics)

    def get_metrics_from_ramdb(self, epoch: int) -> Dict[str, float]:
        """
        Fetch the latest epoch's metrics for the current model.

        Returns:
            Dict[str, float]: A dictionary containing the metrics, where all values are floats.
        """
        sql = """
            SELECT *
            FROM EpochSummary
            WHERE model_id = ? and epoch = ?
            ORDER BY epoch DESC
            LIMIT 1;
        """
        # Pass the parameter correctly as a tuple
        result = self.db.query(sql, params=(self.model_id, epoch), as_dict=True)

        if result:
            return result[0]  # Return the first row as a dictionary
        raise RuntimeError("No records found for the specified model_id")  # Raise error if no records are found






