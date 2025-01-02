from abc import ABC, abstractmethod
from json import dumps
import numpy as np
from numpy import ndarray
from typing import Any
from numpy import array

from src.engine.MgrSQL import MgrSQL
from src.engine.Neuron import Neuron
from src.engine.TrainingData import TrainingData
from datetime import datetime

from src.engine.Utils_DataClasses import Iteration


class Gladiator(ABC):
    def __init__(self,  *args):
        self.gladiator          = args[0]
        self.hyper              = args[1]
        self.training_data      = args[2]                   # Only needed for sqlMgr ==> self.ramDb = args[3]
        self.neurons            = []
        self.neuron_count       = 0                         # Default value
        self.training_data      . reset_to_default()
        self.training_samples   = None                      # To early to get, becaus normalization wouldn't be applied yet self.training_data.get_list()   # Store the list version of training data
        #self.metrics_mgr        = MetricsMgr    (gladiator, self.hyper, self.training_data)
        self.mgr_sql            = MgrSQL        (self.gladiator, self.hyper, self.training_data, self.neurons, args[3]) # Args3, is ramdb
        self._learning_rate     = self.hyper.default_learning_rate
        self.number_of_epochs   = self.hyper.epochs_to_run

    def train(self) -> str:
        if self.neuron_count == 0:
            self.initialize_neurons(1) #Defaults to 1
            print(f"Warning: Defaulting to a single neuron in {self.__class__.__name__}")

        self.training_samples = self.training_data.get_list()           # Store the list version of training data
        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            if epoch % 100 == 0:
                print (f"Epoch: {epoch} for {self.gladiator} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            convg_signal= self.run_an_epoch(epoch)                                # Call function to run single epoch
            if convg_signal !="":                                 # Converged so end early
                return convg_signal
        return "Did not converge"                                         # When it does not converge still return metrics mgr

    def run_an_epoch(self, epoch_num: int) -> bool:             # Function to run single epoch
        for i, sample in enumerate(self.training_samples):         # Loop through all the training data
            sample = np.array(sample)  # Convert sample to a NumPy array
            inputs = sample[:-1]
            target = sample[-1]

            # Capture "before" state
            for neuron in self.neurons:
                neuron.weights_before = np.copy(neuron.weights)
                neuron.bias_before = neuron.bias

            # Step 2: Delegate prediction to the child model
            prediction              = self.training_iteration(sample)  # HERE IS WHERE IT PASSES CONTROL TO THE MODEL BEING TESTED
            error                   = target - prediction
            loss                    = error ** 2  # Example loss calculation

            # Added to support multiple neurons via SQLLite in ram db.
            iteration_data = Iteration(
                model_id=self.mgr_sql.model_id,
                epoch=epoch_num + 1,
                iteration=i + 1,
                inputs=dumps(inputs.tolist()),  # Serialize inputs as JSON
                target=sample[-1],
                prediction=prediction,
                loss=loss,
                accuracy_threshold=self.hyper.accuracy_threshold

            )
            #print("****************************RECORDING ITERATION 1")
            self.mgr_sql.record_iteration(iteration_data)
        return self.mgr_sql.finish_epoch()


    def initialize_neurons(self, neuron_count):
        """
        SNIPER - NOT SHOTGUN
        Spaced values (e.g., equally distributed across
                −
                𝜎
                −σ to
                +
                𝜎
                +σ).
                Scaled eigenvectors of the input data covariance matrix.
                A low-discrepancy sequence (e.g., Sobol or Halton) for quasi-random sampling.
        """
        self.neuron_init = True
        self.neuron_count = neuron_count
        for x in range(neuron_count):                           # Append new neurons to the existing list
            self.neurons.append(Neuron(x, self.training_data.input_count, self.hyper.default_learning_rate))

        '''
            self.neuron_init = True
            self.neuron_count = neuron_count
            self.neurons.clear()                                    # Clear the existing list to start fresh
            for x in range(neuron_count):                           # Append new neurons to the existing list
                self.neurons.append(Neuron(x, self.training_data.input_count, self.hyper.default_learning_rate))
        '''


    @property
    def weights(self):
        """Compatibility attribute pointing to neurons[0].weights."""
        if not self.neurons:
            raise ValueError("No neurons initialized.")
        return self.neurons[0].weights

    @weights.setter
    def weights(self, value):
        """Set weights for neurons[0]."""
        if not self.neurons:
            raise ValueError("No neurons initialized.")
        self.neurons[0].weights = value

    @property
    def bias(self):
        """Compatibility attribute pointing to neurons[0].bias."""
        if not self.neurons:
            raise ValueError("No neurons initialized.")
        return self.neurons[0].bias

    @bias.setter
    def bias(self, value):
        """Set bias for neurons[0]."""
        if not self.neurons:
            raise ValueError("No neurons initialized.")
        self.neurons[0].bias = value

    @property
    def learning_rate(self):
        """Compatibility attribute pointing to neurons[0].learning_rate."""
        if not self.neurons:
            raise ValueError("No neurons initialized.")
        return self.neurons[0].learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        """Set learning rate for neurons[0]."""
        if not self.neurons:
            raise ValueError("No neurons initialized.")
        self.neurons[0].learning_rate = value
