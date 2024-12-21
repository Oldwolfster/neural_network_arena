from abc import ABC, abstractmethod
from src.engine.Metrics import GladiatorOutput, IterationResult, IterationContext
from src.engine.MetricsMgr import MetricsMgr
import numpy as np
from numpy import ndarray
from typing import Any
from numpy import array

from src.engine.Neuron import Neuron
from src.engine.TrainingData import TrainingData
from datetime import datetime

from src.engine.UtilsDataClasses import IterationData, NeuronData


class Gladiator(ABC):
    def __init__(self,  *args):
        gladiator = args[0]
        self.hyper = args[1]
        self.training_data = args[2]
        self.training_data.reset_to_default()
        self.training_samples = None     # To early to get, becaus normalization wouldn't be applied yet self.training_data.get_list()   # Store the list version of training data
        self.metrics_mgr = MetricsMgr(gladiator, self.hyper, self.training_data)
        self._learning_rate = self.hyper.default_learning_rate
        self.number_of_epochs = self.hyper.epochs_to_run
        # WHY? #self.sample_count = len(self.training_samples)          # Calculate and store sample count
        self.neurons = []
        self.neuron_count = 0 # Default value

    def train(self) -> MetricsMgr:
        if self.neuron_count == 0:
            self.initialize_neurons(1) #Defaults to 1
            print(f"Warning: Defaulting to a single neuron in {self.__class__.__name__}")

        self.training_samples = self.training_data.get_list()           # Store the list version of training data
        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            if epoch % 10 == 0 and epoch < 0:
                print (f"Epoch: {epoch} for {self.metrics_mgr.name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if self.run_an_epoch(epoch):                                # Call function to run single epoch
                return self.metrics_mgr                                 # Converged so end early
        return self.metrics_mgr                                         # When it does not converge still return metrics mgr

    def run_an_epoch(self, epoch_num: int) -> bool:             # Function to run single epoch
        for i, sample in enumerate(self.training_samples):         # Loop through all the training data
            sample = np.array(sample)  # Convert sample to a NumPy array
            inputs = sample[:-1]
            target = sample[-1]

            #ORIGINAL PROCESS
            context = IterationContext(             # Record data for this iteration before passing sample to model
                iteration           = i + 1,
                epoch               = epoch_num + 1,
                inputs              = sample[:-1],          # All elements except the last
                target              = sample[-1],           # Last element as target
                weights             = np.copy(self.weights),# Weights PRIOR to model processing Sample
                bias                = self.bias,            # Bias    PRIOR to model processing Sample
                new_weights         = None,
                new_bias            = None
            )

            # Step 1: Capture weights and biases BEFORE processing
            neuron_data_list      = [
                NeuronData(
                    neuron_id       =neuron.nid,
                    inputs=None,  # Will be updated after processing
                    weights=np.copy(neuron.weights),
                    bias=neuron.bias,
                    new_weights=None,  # Placeholder for AFTER weights
                    new_bias=None,  # Placeholder for AFTER bias
                    activation_function="ReLU",  # Placeholder; replace if needed
                    output=None  # Placeholder for output
                )
                for neuron in self.neurons
            ]
            # Step 2: Delegate prediction to the child model
            prediction              = self.training_iteration(sample)  # HERE IS WHERE IT PASSES CONTROL TO THE MODEL BEING TESTED
            error                   = target - prediction
            loss                    = error ** 2  # Example loss calculation

            #ORIGINAL PROCESS
            gladiator_output        = GladiatorOutput(
                prediction          = prediction
            )

            # Step 3: Capture weights and biases AFTER processing
            for neuron, neuron_data in zip(self.neurons, neuron_data_list):
                neuron_data.inputs = inputs  # Update with actual inputs
                neuron_data.new_weights = np.copy(neuron.weights)  # Capture AFTER weights
                neuron_data.new_bias = neuron.bias  # Capture AFTER bias
                neuron_data.output = np.dot(neuron.weights, inputs) + neuron.bias  # Neuron output

            # Step 4: Collect iteration-level data
            iteration_data = IterationData(
                epoch=epoch_num + 1,
                iteration=i + 1,
                inputs=inputs,
                target=target,
                prediction=prediction,
                error=error,
                loss=error ** 2  # Example loss calculation
            )

            #ORIGINAL PROCESS # Put all the information together
            context.new_weights     = np.copy(self.weights) # Weights AFTER model processed sample
            context.new_bias        = self.bias             # Bias    AFTER model processed sample
            result                  = IterationResult(
                gladiator_output    = gladiator_output,
                context             = context,
                # Original above - multi neuron version below
                new_iteration_data  = iteration_data,
                new_neuron_list     = neuron_data_list
            )
            self.metrics_mgr.record_iteration(result)
        return self.metrics_mgr.finish_epoch_summary()

    def initialize_neurons(self, neuron_count):
        self.neuron_init = True
        self.neuron_count = neuron_count
        self.neurons = [Neuron(x, self.training_data.input_count, self.hyper.default_learning_rate) for x in range(neuron_count)]

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
