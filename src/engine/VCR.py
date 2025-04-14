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

from ..Legos.Optimizers import BatchMode


class VCR:       #(gladiator, training_set_size, converge_epochs, converge_threshold, accuracy_threshold, arena_data)  # Create a new Metrics instance with the name as a string
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

        #if (iteration_num) % self.config.batch_size == 0: #maybe rename update weights.
            #self.config.optimizer.finalizer_function(self.config, epoch_num, self.config.gladiator_name)
        #    self.config.optimizer.finalizer_function(self.config.batch_size)
        self.maybe_finalize_batch(iteration_num,   self.training_data.sample_count, self.config.batch_size,  self.config.optimizer.finalizer_function)

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

    def maybe_finalize_batch(self, iteration_num: int, total_samples: int, batch_size: int, finalizer_fn):
        if iteration_num % batch_size == 0:
            # Finalize normal batch
            finalizer_fn(batch_size)
        elif iteration_num == total_samples:
            remainder = total_samples % batch_size
            if remainder > 0:
                finalizer_fn(remainder) # Finalize leftovers

    def finish_epoch(self, epoch: int):
        mae = self.abs_error_for_epoch / self.training_data.sample_count
        if mae < self.config.lowest_error:    # New lowest error
            self.config.lowest_error = mae
            self.config.lowest_error_epoch = epoch

        self.abs_error_for_epoch = 0 # Reset for next epoch
        epoch_metrics = self.get_metrics_from_ramdb(epoch)
        #print(f"VCR ===> MAE = {mae} from dict {epoch_metrics['mean_absolute_error']}")
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

    def record_weight_updates(self, weight_update_metrics):
        """
        Inserts weight update calculations for the current iteration into the database.
        Compatible with arbitrary arg/op chains.
        """
        if not weight_update_metrics:
            return

        sample_row = weight_update_metrics[0]
        fields = self.build_weight_update_field_list(sample_row)
        placeholders = self.build_weight_update_placeholders(sample_row)

        # now to merge it here... self.config.gladiator_name
        sql = f"""
            INSERT INTO WeightAdjustments
            ({fields})
            VALUES ({placeholders})
        """

        table_name = f"WeightAdjustments_{self.config.gladiator_name}" #TODO susceptible to SQL injection
        sql = f"""
            INSERT INTO {table_name}
            ({fields})
            VALUES ({placeholders})
        """

        converted_rows = [self.convert_numpy_scalars_because_python_is_shit(row) for row in weight_update_metrics]
        #print(f"sql={sql}")
        #print(f"converted_rows={converted_rows}")
        self.db.executemany(sql, converted_rows)


    def convert_numpy_scalars_because_python_is_shit(self, row):
        """
        Converts any NumPy scalar values in the given row to their native Python types.
        Friggen ridiculous it was converting either 0 to null or 1 to 0.... what a joke this language is
        """
        return [x.item() if hasattr(x, 'item') else x for x in row]

    def build_weight_update_field_list(self, sample_row):
        base_fields = ["epoch", "iteration", "model_id", "nid", "weight_index", "batch_id"]
        custom_fields = []
        # Now create one custom field per element after the first six (base fields).
        for i in range(6, len(sample_row)):
            arg_n = i - 6 + 1
            custom_fields.append(f"arg_{arg_n}")
        return ", ".join(base_fields + custom_fields)


    def build_weight_update_placeholders(self, sample_row):
        base_placeholders = ["?"] * 6
        arg_op_placeholders = []

        for i in range(6, len(sample_row)):
            arg_op_placeholders.append("CAST(? AS REAL)")  # arg
        return ", ".join(base_placeholders + arg_op_placeholders)




    def record_blame_calculations(self, blame_calculations):
        """
        Inserts all backprop calculations for the current iteration into the database.
        """
        #print("********  Distribute Error Calcs************")
        #for row in self.blame_calculations:
        #    print(row)

        sql = """
        INSERT INTO ErrorSignalCalcs
        (epoch, iteration, model_id, nid, weight_id, 
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
        self.db.executemany(sql, blame_calculations)

