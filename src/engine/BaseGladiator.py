from abc import ABC, abstractmethod
from json import dumps
from src.engine.ActivationFunction import *
from src.engine.MgrSQL import MgrSQL
from src.engine.Neuron import Neuron
from datetime import datetime
from src.engine.Utils_DataClasses import Iteration
from src.engine.WeightInitializer import *
from typing import List


class Gladiator(ABC):
    """
    Abstract base class for training neural network models.

    This class encapsulates the training logic, including forward/backward propagation,
    weight initialization, and convergence detection. It serves as the parent class
    for specific model implementations.
"""
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

    def train(self) -> tuple[str, list[int]]:
        """
        Main method invoked from Framework to train model.

        Returns:
            tuple[str, list[int]]: A tuple containing:
                - converged_signal (str): Which convergence signal(s) triggered early stop.
                - full_architecture (list[int]): Architecture of the model including hidden and output layers.
        """

        if self.neuron_count == 0:
            self.initialize_neurons([]) #Defaults to 1 when it adds the output

        self.training_samples = self.training_data.get_list()           # Store the list version of training data
        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            convg_signal= self.run_an_epoch(epoch)                                # Call function to run single epoch
            if convg_signal !="":                                 # Converged so end early
                return convg_signal, self._full_architecture
        return "Did not converge", self._full_architecture       # When it does not converge still return metrics mgr

    def run_an_epoch(self, epoch_num: int) -> str:
        """
        Executes a training epoch i.e. trains on all samples

        Args:
            epoch_num (int) : number of epoch being executed
        Returns:
            convergence_signal (str) : If not converged, empty string, otherwise signal that detected convergence
        """
        self.epoch = epoch_num      # Set so the child model has access
        if epoch_num % 100 == 0:
                print (f"Epoch: {epoch_num} for {self.gladiator} Loss = {self.last_lost} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for i, sample in enumerate(self.training_samples):  # Loop through all training data
            self.iteration = i      # Set so the model has access
            sample = np.array(sample)  # Convert sample to NumPy array
            inputs = sample[:-1]
            target = sample[-1]
            self.snapshot_weights_as_weights_before()

            # Step 2: Delegate to the model's logic for forward propagation
            self.forward_pass(sample)  # Call model-specific logic
            prediction_raw = Neuron.layers[-1][0].activation_value  # Extract single neuron’s activation

            # Step 3: Delegate to models logic for Validate_pass :)
            error, loss, prediction, loss_gradient = self.validate_pass(prediction_raw, target)
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
        return self.mgr_sql.finish_epoch()      # Finish epoch and return convergence signal

    def snapshot_weights_as_weights_before(self):
        """
         Stores copy of weights and bias for comparison reporting
         """
        for layer_index, current_layer in enumerate(Neuron.layers):
            for neuron in current_layer:
                # Capture "before" state for debugging and backpropagation
                neuron.weights_before = np.copy(neuron.weights)
                neuron.bias_before = neuron.bias

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
        Raises:
            ValueError: If number of initializers is not either 1(apply to all), number of layers, or number of neurons

        """
        total_neurons = sum(architecture)

        # Default to Xavier if not provided
        if not initializers:
            initializers = [Initializer_Xavier] * total_neurons

        # If already one per neuron, assign and return
        if len(initializers) == total_neurons:
            return initializers

        # If one per layer, expand for each neuron in that layer
        if len(initializers) == len(architecture):
            expanded_list = []
            for layer_index, neuron_count in enumerate(architecture):
                expanded_list.extend([initializers[layer_index]] * neuron_count)
            return expanded_list

        # If only one initializer, apply to all neurons
        if len(initializers) == 1:
            expanded_list = [initializers[0]] * total_neurons
            return expanded_list

        # Invalid case: raise error
        raise ValueError(f"Incompatible number of initializers ({len(initializers)}) for total neurons ({total_neurons})")



    def initialize_neurons(self,  architecture: List[int] , initializers: List[WeightInitializer] = None, activation_function_for_hidden: ActivationFunction = Tanh):
        """
        Initializes neurons based on the specified architecture, using appropriate weight initializers.

        Args:
            architecture (List[int]): Number of neurons per hidden layer.
            initializers (List[WeightInitializer]): A list of weight initializers.
            activation_function_for_hidden (ActivationFunction): The activation function for hidden layers.
        """
        if architecture is None:
            architecture = []  # Default to no hidden layers
        architecture.append(1)  # Add output neuron

        # Ensure initializer list matches neuron count
        flat_initializers = self.get_flat_initializers(architecture, initializers)
        print(f"Checking flat_initializers: {flat_initializers}")

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

                activation = output_af if layer_index==len(architecture)-1 else activation_function_for_hidden
                #print(f"Creating Neuron {nid}  in layer{layer_index}  len(architecture)={len(architecture)} - Act = {activation.name}")
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

    @property
    def learning_rate(self):
        """
        Getter for learning rate.
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, new_learning_rate: float):
        """
        Updates the learning rate for the Gladiator and ensures all neurons reflect the change.

        Args:
            new_learning_rate (float): The new learning rate to set.
        """
        self._learning_rate = new_learning_rate
        for neuron in self.neurons:
            neuron.learning_rate = new_learning_rate
