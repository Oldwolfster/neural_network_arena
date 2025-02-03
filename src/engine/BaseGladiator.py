from abc import ABC, abstractmethod
from json import dumps

import numpy
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
        #self.layers             = []                        # Layered structure
        self.neuron_count       = 0                         # Default value
        self.training_data      . reset_to_default()
        self.training_samples   = None                      # To early to get, becaus normalization wouldn't be applied yet self.training_data.get_list()   # Store the list version of training data
        self.mgr_sql            = MgrSQL(self.gladiator, self.hyper, self.training_data, self.neurons, args[3]) # Args3, is ramdb
        self._learning_rate     = self.hyper.default_learning_rate #todo set this to all neurons learning rate
        self.number_of_epochs   = self.hyper.epochs_to_run
        self.full_architecture  = None
        self.last_lost          = 0
        self.iteration          = 0
        self.epoch              = 0

    def train(self) -> str:
        if self.neuron_count == 0:
            self.initialize_neurons([]) #Defaults to 1 when it adds the output
            print(f"Warning: Defaulting to a single neuron in {self.__class__.__name__}")

        self.training_samples = self.training_data.get_list()           # Store the list version of training data
        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            self.epoch = epoch      # Set so the model has access
            if epoch % 100 == 0:
                print (f"Epoch: {epoch} for {self.gladiator} Loss = {self.last_lost} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            convg_signal= self.run_an_epoch(epoch)                                # Call function to run single epoch
            if convg_signal !="":                                 # Converged so end early
                return convg_signal, self.full_architecture
        print (f"FULL ARCHITECTURE DEFINITION={self.full_architecture}")
        return "Did not converge", self.full_architecture       # When it does not converge still return metrics mgr

    def run_an_epoch(self, epoch_num: int) -> bool:
        for i, sample in enumerate(self.training_samples):  # Loop through all training data
            self.iteration = i      # Set so the model has access
            sample = np.array(sample)  # Convert sample to NumPy array
            inputs = sample[:-1]
            target = sample[-1]
            #self.run_forward_pass(inputs)
            self.snapshot_weights_as_weights_before(inputs)
            # Step 2: Delegate to the model's logic for forward propagation
            model_style=""
            if hasattr(self, 'training_iteration') and callable(self.training_iteration):
                model_style="Old"
            elif hasattr(self, 'forward_pass') and callable(self.forward_pass):
                model_style="New"
            else:
                raise NotImplementedError("Subclass must implement either run_iteration or run_forward_pass")
            prediction_raw = 0.0
            if model_style == "Old":
                prediction_raw = self.training_iteration(sample)
            else:
                self.forward_pass(sample)  # Call model-specific logic
                # Step 3: Validate_pass :)
                prediction_raw = Neuron.layers[-1][0].activation_value  # Extract single neuron’s activation

            error = target - prediction_raw
            loss = error ** 2  # Example loss calculation (MSE for a single sample)
            prediction =  1 if prediction_raw > 0 else 0      # Apply step function
            loss_gradient = error * 2 #For MSE it is linear.
            self.last_lost = loss

            if model_style=="New":
                self.back_pass(sample, loss_gradient)  # Call model-specific logic

            # Step 4: Delegate to models logic for backporop.
            #self.backwards_pass(sample)  # Call model-specific logic

            #print(f"prediction_raw={prediction_raw}\ttarget={target}\terror={error}")
            # Step 4: Record iteration data
            iteration_data = Iteration(
                model_id=self.mgr_sql.model_id,
                epoch=epoch_num + 1,
                iteration=i + 1,
                inputs=dumps(inputs.tolist()),  # Serialize inputs as JSON
                target=sample[-1],
                prediction=prediction,
                prediction_raw=prediction_raw,
                loss=loss,
                loss_gradient=loss_gradient,
                accuracy_threshold=self.hyper.accuracy_threshold,
            )

            self.mgr_sql.record_iteration(iteration_data, Neuron.layers)

        # Finish epoch and return convergence signal
        return self.mgr_sql.finish_epoch()

    def snapshot_weights_as_weights_before(self, inputs: numpy.array):
        """
        Executes the forward propagation for one sample, ensuring weights_before are correctly captured.
        """

        for layer_index, current_layer in enumerate(Neuron.layers):
            if layer_index == 0:
                # First hidden layer: Takes inputs directly from the sample
                prev_values = inputs
            else:
                # Subsequent layers: Takes inputs from previous layer's activations
                prev_values = [neuron.activation_value for neuron in Neuron.layers[layer_index - 1]]

            for neuron in current_layer:
                # Capture "before" state for debugging and backpropagation
                neuron.weights_before = np.copy(neuron.weights)
                neuron.bias_before = neuron.bias

                # Compute weighted sum and activation
                neuron.raw_sum = sum(weight * value for weight, value in zip(neuron.weights, prev_values)) + neuron.bias
                neuron.activation_value = neuron.raw_sum  # TODO: Apply activation function

            # Debugging
            # print(f"DEBUG Layer {layer_index} Activations: {[n.activation_value for n in current_layer]}")

    def initialize_neurons(self, architecture: list = None):
        """
        Initializes neurons based on the specified architecture.

        Args:
            architecture (list): List of integers representing the number of neurons
                                 in each hidden layer.
                                 Example: [4, 3, 2] for 4 neurons in the first hidden layer,
                                 3 in the next, and 2 in the final hidden layer.
                                 The output neuron is automatically added.
        """
        if architecture is None:
            architecture = []  # Default to no hidden layers, just a single output neuron
        architecture.append(1)  # Append output neuron (force single-output)
        input_count = self.training_data.input_count
        self.full_architecture = [input_count] + architecture # Store the full architecture

        # Initialize neurons for all layers except the input layer
        self.neurons.clear()  # Clear any existing neurons
        Neuron.layers.clear()   # Clear any existing Layered structure
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
                #layer_neurons.append(neuron)    # Add to current layer
            #Neuron.layers.append(layer_neurons)   # Add current layer to layered structure
        self.neuron_count = len(self.neurons)
        #print(f"*******IN initialize_neurons Neurons: {len(self.neurons)} architecture = {architecture}")
        #for i, neuron in enumerate(self.neurons):
        #    print(f"Neuron i={i}\tnid={neuron.nid}: \tneuron.layer_id = {neuron.layer_id} Weights = {neuron.weights}, Bias = {neuron.bias}")
        #self.print_layer_debug_info()
    def print_layer_debug_info(self):
        for layer_index in range(len(self.layers)):
            print(f"layer ================== {layer_index}")
            for neuron in self.layers[layer_index]:
                print (f"nid={neuron.nid}\tweights={neuron.weights_before}\tinputs={neuron.neuron_inputs}")




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
        """ Getter for learning rate. """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, new_learning_rate: float):
        """
        Updates the learning rate for the Gladiator
        and ensures all neurons reflect the change.
        """
        self._learning_rate = new_learning_rate
        for neuron in self.neurons:
            neuron.learning_rate = new_learning_rate
