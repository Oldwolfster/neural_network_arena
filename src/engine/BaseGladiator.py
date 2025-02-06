from abc import ABC, abstractmethod
from json import dumps
import random
import numpy
from numpy import ndarray
from typing import Any
from numpy import array

from src.engine.ActivationFunction import *
from src.engine.MgrSQL import MgrSQL
from src.engine.Neuron import Neuron
from src.engine.TrainingData import TrainingData
from datetime import datetime
from src.engine.Utils_DataClasses import Iteration
from src.engine.WeightInitializer import *
from typing import List


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
        self._full_architecture  = None
        self.last_lost          = 0
        self.iteration          = 0
        self.epoch              = 0
        self.neuron_initializers= []        # Delegate functions to init neurons
        #if self.hyper.random_seed == 0:
        #    self.random_seed        = random.randint(1, 999999)
        #else:
        #    self.random_seed = self.hyper.random_seed
        #self.set_random_seed(self.random_seed) # 🔹 Set the seed for reproducibility

    def set_random_seed(self, seed):

        np.random.seed(seed)
        random.seed(seed)

    def train(self) -> str:
        if self.neuron_count == 0:
            self.initialize_neurons([]) #Defaults to 1 when it adds the output

        self.training_samples = self.training_data.get_list()           # Store the list version of training data
        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            convg_signal= self.run_an_epoch(epoch)                                # Call function to run single epoch
            if convg_signal !="":                                 # Converged so end early
                return convg_signal, self._full_architecture
        return "Did not converge", self._full_architecture       # When it does not converge still return metrics mgr

    def run_an_epoch(self, epoch_num: int) -> bool:
        self.epoch = epoch_num      # Set so the child model has access
        if epoch_num % 100 == 0:
                print (f"Epoch: {epoch_num} for {self.gladiator} Loss = {self.last_lost} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for i, sample in enumerate(self.training_samples):  # Loop through all training data
            self.iteration = i      # Set so the model has access
            sample = np.array(sample)  # Convert sample to NumPy array
            inputs = sample[:-1]
            target = sample[-1]
            self.snapshot_weights_as_weights_before(inputs)

            # Step 2: Delegate to the model's logic for forward propagation
            self.forward_pass(sample)  # Call model-specific logic
            prediction_raw = Neuron.layers[-1][0].activation_value  # Extract single neuron’s activation

            # Step 3: Validate_pass :)
            error = target - prediction_raw
            loss = error ** 2  # Example loss calculation (MSE for a single sample)
            prediction =  1 if prediction_raw > .5 else 0      # Apply step function
            loss_gradient = error * 2 #For MSE it is linear.
            self.last_lost = loss

            # Step 4: Delegate to models logic for backporop.
            self.back_pass(sample, loss_gradient)  # Call model-specific logic
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
        """        Copies weights and bias        """
        for layer_index, current_layer in enumerate(Neuron.layers):
            for neuron in current_layer:
                # Capture "before" state for debugging and backpropagation
                neuron.weights_before = np.copy(neuron.weights)
                neuron.bias_before = neuron.bias


    def snapshot_weights_as_weights_before_doing_way_to_muchDeleteMe(self, inputs: numpy.array):
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


    def get_flat_initializers(self, architecture: List[int], initializers: List[WeightInitializer]) -> List[WeightInitializer]:
        """
        Returns a flat list of weight initializers, ensuring compatibility with the number of neurons.
        If neuron_initializers is not set correctly, this method updates it.

        Cases handled:
          - If empty → Defaults to Xavier for all neurons
          - If matches neuron count → Use as-is
          - If matches layer count → Expand per neuron
          - If has only one initializer → Expand for all neurons
          - If inconsistent → Raise error

        Args:
            architecture (List[int]): The network's architecture (neurons per layer)

        Returns:
            List[WeightInitializer]: A list of initializers for each neuron.
        """
        total_neurons = sum(architecture)

        # Default to Xavier if not set
        #if not self.neuron_initializers:
        if not initializers:
            initializers = [Initializer_Xavier] * total_neurons

        # If already correct, return as-is
        if len(initializers) == total_neurons:
            return self.neuron_initializers

        # If matches layer count, expand per neuron
        if len(initializers) == len(architecture):
            expanded_list = []
            for layer_index, neuron_count in enumerate(architecture):
                expanded_list.extend([self.neuron_initializers[layer_index]] * neuron_count)
            self.neuron_initializers = expanded_list
            return self.neuron_initializers

        # If only one initializer, apply to all neurons
        if len(initializers) == 1:
            initializers = [initializers[0]] * total_neurons
            self.neuron_initializers = initializers
            return self.neuron_initializers

        # Invalid case: raise error
        raise ValueError(f"Incompatible number of initializers ({len(self.neuron_initializers)}) "
                         f"for total neurons ({total_neurons})")

    def initialize_neurons(self,  architecture: List[int] , initializers: List[WeightInitializer] = None, default_AF: ActivationFunction = Tanh):
        """
        Initializes neurons based on the specified architecture, using appropriate weight initializers.

        Args:
            architecture (List[int]): Number of neurons per hidden layer.
            default_AF (Activation Function): Sets AF for all neurons except output layer
        """
        if architecture is None:
            architecture = []  # Default to no hidden layers
        architecture.append(1)  # Add output neuron

        # Ensure initializer list matches neuron count
        flat_initializers = self.get_flat_initializers(architecture, initializers)

        input_count = self.training_data.input_count
        self._full_architecture = [input_count] + architecture  # Store the full architecture

        self.neurons.clear()
        Neuron.layers.clear()
        nid = -1
        output_af = Sigmoid #Assume binary decision
        if self.training_data.problem_type !="Binary Decision":
            output_af = Linear

        for layer_index, neuron_count in enumerate(architecture):
            num_of_weights = self._full_architecture[layer_index]
            for neuron_index in range(neuron_count):
                nid += 1

                activation = output_af if layer_index==len(architecture)-1 else default_AF
                print(f"Creating Neuron {nid}  in layer{layer_index}  len(architecture)={len(architecture)} - Act = {activation.name}")
                neuron = Neuron(
                    nid=nid,
                    num_of_weights=num_of_weights,
                    learning_rate=self.hyper.default_learning_rate,
                    weight_initializer=flat_initializers[nid],  # Assign correct initializer
                    layer_id=layer_index,
                    activation=activation
                )
                self.neurons.append(neuron)
        self.neuron_count = len(self.neurons)



    """
        def training_iteration(self):
            model_style=""
            if hasattr(self, 'training_iteration') and callable(self.training_iteration):
                model_style="Old"
            elif hasattr(self, 'forward_pass') and callable(self.forward_pass):
                model_style="New"
            else:
                raise NotImplementedError("Subclass must implement either run_iteration or run_forward_pass")
    """

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
