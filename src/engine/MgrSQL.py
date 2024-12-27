import sqlite3
from json import dumps
from typing import List

from .Utils import IterationResult
from src.ArenaSettings import HyperParameters
from .TrainingData import TrainingData
from .Utils_DataClasses import IterationData


class MgrSQL:       #(gladiator, training_set_size, converge_epochs, converge_threshold, accuracy_threshold, arena_data)  # Create a new Metrics instance with the name as a string
    def __init__(self, model_id, hyper: HyperParameters, training_data: TrainingData, neurons: List, ramDb: sqlite3.Connection):
        # Run Level members
        self.training_data          = training_data
        self.model_id               = model_id
        self.hyper                  = hyper
        self.neurons                = neurons
        self.ramDb                  = ramDb
                                                                #REMOVED WHEN ADDING SQL self.epoch_summaries = []           # Store the epoch summaries
                                                                #REMOVED WHEN ADDING SQL self.summary = EpochSummary()       # The current Epoch summary
        self.run_time               = 0                         # How long did training take?
        self.iteration_num          = 0                         # Current Iteration #
        self.epoch_curr_number      = 1                         # Which epoch are we currently on.
        self.sample_count           = len(self.training_data.get_list())          # Calculate and store sample count= 0               # Number of samples in each iteration.
        self.accuracy_threshold     = (hyper.accuracy_threshold)    # In regression, how close must it be to be considered "accurate"
                                                                    #REMOVED WHEN ADDING SQL Metrics.set_acc_threshold(hyper.accuracy_threshold)    # Set at Class level (not instance) one value shared across all instances
                                                                    #REMOVED WHEN ADDING SQL self.metrics = []                   # The list of metrics this manager is running.
        self._converge_detector = None
        self.convergence_signal = []      # Will be set by convergence detector

        #Below added when switching to multiple neurons
        self.metrics_iteration = []                                 # Accumulate iteration data for batch writes
        self.metrics_neuron = []                                    # Accumulate iteration data for batch writes


    @property
    def converge_detector(self):
        """
        Provides lazy instantiation of converge_detector so it can pass it(CD) a copy of itself (MMgr)
        """
        if self._converge_detector is None:
            # Lazy import to avoid circular reference
            from src.engine.convergence.ConvergenceDetector import ConvergenceDetector
            self._converge_detector = ConvergenceDetector(self.hyper, self)
        return self._converge_detector

    def record_iteration(self, iteration_data: IterationData):
        print("****************************RECORDING ITERATION 2")
        self.metrics_iteration.append(iteration_data)

        for neuron in self.neurons:
            self.record_neuron(iteration_data, neuron)

    def record_neuron(self, iteration, neuron):
        """
        Add a single neuron's data to the accumulator for the current iteration.
        """
        print("****************************RECORDING ITERATION 3")
        print(f"recording neuron {neuron.nid}")
        self.metrics_neuron.append({
            'model_id': self.model_id,
            'epoch': iteration.epoch,
            'step': iteration.step,
            'neuron_id': neuron.nid,
            'layer_id': neuron.layer_id,
            'weights': dumps(neuron.weights.tolist()),  # Serialize weights
            'bias': neuron.bias,
            'output': neuron.output,
        })

    def finish_epoch(self):
        self.write_iterations()
        self.write_neurons()


    def write_neurons(self):
        """
        Batch write all accumulated neuron data to the database.
        """
        if not self.metrics_neuron:
            return  # Nothing to write
        sql_neurons = '''
            INSERT INTO neurons (model_id, epoch, step, neuron_id, layer_id, weights, bias, output)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        values_neurons = [
            (
                neuron['model_id'],
                neuron['epoch'],
                neuron['step'],
                neuron['neuron_id'],
                neuron['layer_id'],
                neuron['weights'],
                neuron['bias'],
                neuron['output']
            )
            for neuron in self.metrics_neuron
        ]
        self.ramDb.executemany(sql_neurons, values_neurons)
        self.metrics_neuron.clear()


    def write_iterations(self):
        """
        Batch write all accumulated iteration data to the database.
        """
        if not self.metrics_iteration:
            return  # Nothing to write

        sql = '''
            INSERT INTO iterations (model_id, epoch, step, inputs, target, prediction, loss)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        values = [
            (
                iteration.model_id,
                iteration.epoch,
                iteration.step,
                iteration.inputs,
                iteration.target,
                iteration.prediction,
                iteration.loss
            )
            for iteration in self.metrics_iteration
        ]
        self.ramDb.executemany(sql, values)
        self.metrics_iteration.clear()  # Clear the accumulator after writing

