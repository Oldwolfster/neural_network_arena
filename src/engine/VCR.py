import json
import sqlite3
from json import dumps
from typing import List

import numpy as np

from .Neuron import Neuron
from .RamDB import RamDB

from src.ArenaSettings import HyperParameters
from .TrainingData import TrainingData
from .TrainingRunInfo import TrainingRunInfo
from .Utils_DataClasses import Iteration, RecordLevel
from src.engine.convergence.ConvergenceDetector import ConvergenceDetector
from typing import Dict

from ..Legos.Optimizers import BatchMode


class VCR:
    def __init__(self, TRI, neurons: List):
        # Run Level members
        self.TRI : TrainingRunInfo  = TRI
        self.neurons                = neurons
        self.batch_id               = 0
        self.iteration_num          = 0                         # Current Iteration #
        self.epoch_curr_number      = 1                         # Which epoch are we currently on.
        self.sample_count           = len(TRI.training_data.get_list())          # Calculate and store sample count= 0               # Number of samples in each iteration.
        self.converge_detector      = ConvergenceDetector(TRI.hyper, TRI.training_data, TRI.config)
        self.abs_error_for_epoch    = 0
        self.bd_correct             = 0
        self.convergence_signal     = None      # Will be set by convergence detector
        self.backpass_finalize_info = []

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
        self.abs_error_for_epoch += abs(iteration_data.error_unscaled)
        if iteration_data.is_true: self.bd_correct += 1

        self.bd_correct
        record_weight_updates_from_finalize = self.maybe_finalize_batch(iteration_data.iteration,   self.TRI.training_data.sample_count, self.TRI.config.batch_size,  self.TRI.config.optimizer.finalizer)

        if any(record_weight_updates_from_finalize):
            self.record_weight_updates(record_weight_updates_from_finalize, "finalize")

        if not self.TRI.should_record(RecordLevel.FULL ): return
        self.TRI.db.add(iteration_data)

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

                # Add the neuron data to the database
                self.TRI.db.add(neuron, exclude_keys={"activation", "learning_rate", "weights", "weights_before"}, run_id=self.TRI.run_id, epoch_n=iteration_data.epoch, iteration_n=iteration_data.iteration)
        Neuron.bulk_insert_weights(db = self.TRI.db, run_id = self.TRI.run_id, epoch=iteration_data.epoch, iteration=iteration_data.iteration )

    def maybe_finalize_batch(self, iteration_num: int, total_samples: int, batch_size: int, finalizer_fn) -> list:
        if iteration_num % batch_size == 0:
            # replaced with the below to standardize batch_id handling return finalizer_fn(batch_size, self.epoch_curr_number, iteration_num)  # Normal batch
            return self.finish_batch(batch_size,iteration_num,finalizer_fn)
        elif iteration_num == total_samples:
            remainder = total_samples % batch_size
            if remainder > 0:
                # replaced with the below to standardize batch_id handlingreturn finalizer_fn(remainder, self.epoch_curr_number, iteration_num)  # Final mini-batch
                return self.finish_batch(remainder,iteration_num,finalizer_fn)
        return []  # Nothing to finalize this round

    def finish_batch(self, batch_size, iteration_num, finalizer_fn) -> list:
        """
            Runs the optimizer's finalizer function with the correct batch_id,
            and increments the internal batch counter for the next batch.
            This ensures all finalizers remain stateless and batch_id is standardized.
        """
        finalizer_log = finalizer_fn(batch_size, self.epoch_curr_number, iteration_num, self.batch_id)  # Normal batch
        self.batch_id += 1    #increment batch number
        return finalizer_log

    def finish_epoch(self, epoch: int):
        self.TRI.last_epoch = epoch
        mae = self.abs_error_for_epoch / self.TRI.training_data.sample_count
        #print(f"mae={mae}")
        self.TRI.set("mae", mae)
        if self.TRI.set_if_lower("lowest_mae", mae):
            self.TRI.set("lowest_error_epoch", epoch)
        self.TRI.set("bd_correct", self.bd_correct)
        self.abs_error_for_epoch = 0                        # Reset for next epoch
        self.bd_correct = 0                                 # Reset for next epoch
        self.epoch_curr_number += 1
        val =  self.converge_detector.check_convergence(self.epoch_curr_number, mae )
        return val


    ############# Record Backpass info for pop up window of NeuroForge #############
    ############# Record Backpass info for pop up window of NeuroForge #############
    ############# Record Backpass info for pop up window of NeuroForge #############
    ############# Record Backpass info for pop up window of NeuroForge #############
    def record_weight_updates(self, weight_update_metrics, update_or_finalize: str):
        """
        Inserts weight update calculations for the current iteration into the database.
        Compatible with arbitrary arg/op chains.
        """
        if not self.TRI.should_record(RecordLevel.FULL ): return
        sample_row = weight_update_metrics[0]
        fields = self.build_weight_update_field_list(sample_row)
        placeholders = self.build_weight_update_placeholders(sample_row)

        table_name = f"WeightAdjustments_{update_or_finalize}_{self.TRI.run_id}" #TODO susceptible to SQL injection
        sql = f"""
            INSERT INTO {table_name}
            ({fields})
            VALUES ({placeholders})
        """

        converted_rows = [self.convert_numpy_scalars_because_python_is_shit(row) for row in weight_update_metrics]

        #print(f"Data about to be  INSERT{converted_rows}")
        #print("Below is table content")
        #self.TRI.db.query_print(f"Select * from {table_name}")

        self.TRI.db.executemany(sql, converted_rows,"weight adjustments")
        #print("Insert worked")
        weight_update_metrics.clear()

    def convert_numpy_scalars_because_python_is_shit(self, row):
        """
        Converts any NumPy scalar values in the given row to their native Python types.
        Friggen ridiculous it was converting either 0 to null or 1 to 0.... what a joke this language is
        """
        return [x.item() if hasattr(x, 'item') else x for x in row]

    def build_weight_update_field_list(self, sample_row):
        #base_fields = ["epoch", "iteration", "model_id", "nid", "weight_index", "batch_id"]
        base_fields = ["epoch", "iteration", "nid", "weight_index", "batch_id"]
        custom_fields = []
        # Now create one custom field per element after the first six (base fields).
        for i in range(5, len(sample_row)):
            arg_n = i - 5 + 1
            custom_fields.append(f"arg_{arg_n}")
        return ", ".join(base_fields + custom_fields)

    def build_weight_update_placeholders(self, sample_row):
        #base_placeholders = ["?"] * 6
        base_placeholders = ["?"] * 5
        arg_op_placeholders = []

        #for i in range(6, len(sample_row)):
        for i in range(5, len(sample_row)):
            arg_op_placeholders.append("CAST(? AS REAL)")  # arg
        return ", ".join(base_placeholders + arg_op_placeholders)

    def record_blame_calculations(self, blame_calculations):
        """
        Inserts all backprop calculations for the current iteration into the database.
        """

        #print("********  Distribute Error Calcs************")
        #for row in self.blame_calculations:
        #    print(row)
        if not self.TRI.should_record(RecordLevel.FULL ): return
        sql = """
        INSERT INTO ErrorSignalCalcs
        (epoch, iteration, run_id, nid, weight_id, 
         arg_1, op_1, arg_2, op_2, arg_3, op_3, result)
        VALUES 
        (?, ?, ?, ?, ?, 
         CAST(? AS REAL), ?, 
         CAST(? AS REAL), ?, 
         CAST(? AS REAL), ?, 
         CAST(? AS REAL))
        """

        # Convert each row to ensure any numpy scalars are native Python types
        converted_rows = [self.convert_numpy_scalars_because_python_is_shit(row) for row in blame_calculations]
        #print(f"BLAME {self.blame_calculations}")

        #Heads up, sometimes overflow error look like key violation here
        self.TRI.db.executemany(sql, blame_calculations, "error signal")
        blame_calculations.clear()

