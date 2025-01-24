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
        self.layers             = []                        # Layered structure
        self.neuron_count       = 0                         # Default value
        self.training_data      . reset_to_default()
        self.training_samples   = None                      # To early to get, becaus normalization wouldn't be applied yet self.training_data.get_list()   # Store the list version of training data
        self.mgr_sql            = MgrSQL(self.gladiator, self.hyper, self.training_data, self.neurons, args[3]) # Args3, is ramdb
        self._learning_rate     = self.hyper.default_learning_rate
        self.number_of_epochs   = self.hyper.epochs_to_run
        self.full_architecture  = None

    def train(self) -> str:
        if self.neuron_count == 0:
            self.initialize_neurons([1]) #Defaults to 1
            print(f"Warning: Defaulting to a single neuron in {self.__class__.__name__}")

        self.training_samples = self.training_data.get_list()           # Store the list version of training data
        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            if epoch % 100 == 0:
                print (f"Epoch: {epoch} for {self.gladiator} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            convg_signal= self.run_an_epoch(epoch)                                # Call function to run single epoch
            if convg_signal !="":                                 # Converged so end early
                return convg_signal, self.full_architecture
        print (f"FULL ARCHITECTURE DEFINITION={self.full_architecture}")
        return "Did not converge", self.full_architecture       # When it does not converge still return metrics mgr

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

    def initialize_neurons(self, architecture: list = None):
        """
        Initializes neurons based on the specified architecture.

        Args:
            architecture (list): List of integers representing the number of neurons
                                 in each hidden/output layer.
                                 Example: [4, 3, 2, 1] for 4 neurons in the first hidden layer,
                                 3 in the next, etc.
        """
        # Ensure architecture is defined
        if architecture is None:
            architecture = [1]  # Default to a single neuron in a single layer

        # Combine inputs and architecture into the full architecture
        input_count = self.training_data.input_count
        print (f"********** in base gladiator input count={input_count}")
        self.full_architecture = [input_count] + architecture  # Store the full architecture

        # Initialize neurons for all layers except the input layer
        self.neurons.clear()  # Clear any existing neurons
        self.layers.clear()   # Clear any existing Layered structure
        nid = -1
        for layer_index, neuron_count in enumerate(architecture):
            layer_neurons = []  # Temporary list for current layer
            num_of_weights = self.full_architecture[layer_index]
            #print(f"in BaseGladiator layer_index = {layer_index}\tself.full_architecture = {self.full_architecture}\tnum_of_weights = {num_of_weights}")

            for neuron_index in range(neuron_count):
                nid += 1
                neuron = Neuron(
                    nid=nid,
                    num_of_weights=num_of_weights,
                    learning_rate=self.hyper.default_learning_rate,
                    layer_id=layer_index
                )
                self.neurons.append(neuron)     # Add to flat list
                layer_neurons.append(neuron)    # Add to current layer
            self.layers.append(layer_neurons)   # Add current layer to layered structure
        self.neuron_count = len(self.neurons)
        #print(f"IN initialize_neurons Neurons: {len(self.neurons)} architecture = {architecture}")
        #for i, neuron in enumerate(self.neurons):
        #    print(f"Neuron {i}: Weights = {neuron.weights}, Bias = {neuron.bias}")

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
