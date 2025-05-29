from abc import ABC
from json import dumps

import numpy as np

from src.Legos.ActivationFunctions import *
from src.Legos.Optimizers import *
from src.engine.TrainingRunInfo import TrainingRunInfo
from src.engine.VCR import VCR
from src.engine.Config import Config
from src.engine.Neuron import Neuron
from datetime import datetime
from src.engine.Utils_DataClasses import ez_debug

from src.engine.Utils_DataClasses import Iteration
from src.Legos.WeightInitializers import *
from typing import List, Tuple

"""

"""

class Gladiator(ABC):
    """
    💥 NOTE: The gradient is inverted from the traditional way of thinking.
    Abstract base class for creating Gladiators (neural network models).
    Goal: Give child gladiator class as much power as possible, requiring as little responsibility
    as possible, while allowing for overwriting any step of the process.

    There are three main sections:
    1) Training Default Methods - available for free, but overwritable for experimentation.
    2) Training Framework - does the brute force tasks of the arena - not intended for overriding
    3) Initialization - Preps everything for the framework and gladiator
"""
    #def __init__(self,  config: Config):
    def __init__(self,  TRI: TrainingRunInfo):
        self.TRI                = TRI # TrainingRunInfo
        self.config             = TRI.config
        self.db                 = TRI.db
        self.hyper              = TRI.hyper
        self.training_data      = TRI.training_data              # Only needed for sqlMgr ==> self.ramDb = args[3]
        self.total_iterations   = 1                         # Timestep for optimizers such as adam
        #self.training_samples   = None                      # To early to get, becaus normalization wouldn't be applied yet self.training_data.get_list()   # Store the list version of training data
        self.VCR                = VCR(TRI, Neuron.neurons) # Args3, is ramdb
        #self.learning_rate      = self.config.learning_rate  #
        self.number_of_epochs   = self.hyper.epochs_to_run
        #self._full_a1rchitecture = None
        self._bd_threshold       = None
        self._bd_class_alpha     = None
        self._bd_class_beta      = None
        self.total_error_for_epoch          = 0
        self.iteration          = 0
        self.epoch              = 0
        self.too_high_adjst     = self.training_data.input_max * 5 #TODO make 5 hyperparamter
        self.max_adj            = self.training_data.everything_max_magnitude()
        self.blame_calculations = []
        self.weight_update_calculations = []
        self.convergence_phase  = "watch"
        self.finalize_setup()

    ################################################################################################
    ################################ SECTION 1 - pipeline ####################################
    ################################################################################################

    def finalize_setup(self):
        self.configure_model(self.config)  #Typically overwritten in child  class.

        self.config.smartNetworkSetup(self.TRI.setup)

        self.initialize_neurons(
            architecture=self.config.architecture.copy(),  # Avoid mutation
            initializers=[self.config.initializer],  # <- List of 1 initializer
            hidden_activation=self.config.hidden_activation,
            output_activation=self.config.output_activation
            or self.config.loss_function.recommended_output_activation)

        #print(f"Neuron count = {len(Neuron.neurons)}")
        self.customize_neurons(self.config)

    def configure_model(self, config: Config): #Typically overwritten in child  class.
        pass

    def customize_neurons(self,config: Config): #Typically overwritten in child  class.
        pass

    def scale_samples(self):  #Scales the inputs and targets according to the config in the model and config defaults
        self.config.scaler.scale_all()

    def train(self, exploratory_epochs = 0):  #tuple[str, list[int]]:
        """
        Main method invoked from Framework to train model.

        Parameters:
            exploratory_epochs: If doing a LR sweep or something, how many epochs to check.
        Returns:
            tuple[str, list[int]]: A tuple containing:
                - converged_signal (str): Which convergence signal(s) triggered early stop.
                - full_architecture (list[int]): Architecture of the model including hidden and output layers.
        """

        self.config.loss_function.validate_activation_functions()
        self.scale_samples()
        epochs_to_run = self.number_of_epochs if exploratory_epochs == 0 else exploratory_epochs

        for epoch in range(epochs_to_run):                              # Loop to run specified # of epochs
            self.config.final_epoch = epoch                             # If correct, great, if not, corrected next epoch.
            self.config.cvg_condition = self.run_an_epoch(epoch)        # Call function to run single epoch
            self.TRI.set("total_error_for_epoch",self.total_error_for_epoch)
            if self.config.cvg_condition    != "Did Not Converge":         # Converged so end early
                return


    def run_an_epoch(self, epoch_num: int) -> str:
        """
        Executes a training epoch i.e. trains on all samples

        Args:
            epoch_num (int) : number of epoch being executed
        Returns:
            convergence_signal (str) : If not converged, empty string, otherwise signal that detected convergence
        """

        self.epoch = epoch_num      # Set so the child model has access
        if epoch_num % 100 == 0 and epoch_num!=0:  print (f"Epoch: {epoch_num} for {self.config.gladiator_name} Loss = {self.last_epoch_mae} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.total_error_for_epoch = 0

        for self.iteration, (sample, sample_unscaled) in enumerate(zip(self.config.scaler.scaled_samples, self.config.scaler.unscaled_samples)):
            self.run_a_sample(np.array(sample), np.array(sample_unscaled))
            if self.total_error_for_epoch > 1e30:   return  "Gradient Explosion" #Check for gradient explosion
        return self.VCR.finish_epoch(epoch_num + 1)      # Finish epoch and return convergence signal

    def run_a_sample(self, sample, sample_unscaled):
        self                . snapshot_weights("", "_before")
        error, loss, blame  = self.optimize_passes(sample)
        self                . total_error_for_epoch += abs(error)

         # 3 possible prediction values... raw, unscaled, and thresholded(called just prediction) for binary decision
        #TODO optimize pass also has prediction_raw... seems there should only be one.
        prediction_raw      = Neuron.output_neuron.activation_value  # Extract single neuron’s activation
        prediction          = self.config.threshold_prediction(prediction_raw)

        #print(f"target unscaled\t{sample_unscaled[-1]}")
        #print(f"prediction unscaled\t{self.config.scaler.unscale_target(prediction_raw)}")
        #print(" ")



        # Step 4: Record iteration data
        iteration_data = Iteration(
            model_id=self.config.gladiator_name,
            epoch=self.epoch + 1,
            iteration=self.iteration + 1,
            inputs=dumps(sample[:-1].tolist()),  # Serialize inputs as JSON
            inputs_unscaled=dumps(sample_unscaled[:-1].tolist()),  # Serialize inputs as JSON
            target=sample[-1],
            target_unscaled=sample_unscaled[-1],
            prediction=prediction,
            prediction_unscaled=self.config.scaler.unscale_target(prediction_raw),#converts prediction back to "unscaled space"
            prediction_raw=prediction_raw,
            loss=loss,
            loss_function=self.config.loss_function.name,
            loss_gradient=blame,
            accuracy_threshold=self.hyper.accuracy_threshold,
        )
        self.VCR.record_iteration(iteration_data, Neuron.layers)

    # noinspection PyTypeChecker
    def optimize_passes(self, sample):
        # Step 1: Forward pass
        prediction_raw = self.forward_pass(sample)  # Call model-specific logic
        if prediction_raw is None: raise ValueError(f"{self.__class__.__name__}.forward_pass must return a value for sample={sample!r}"            ) # ensure forward_pass actually returned something

        # Step 2: Judge pass - calculate error, loss, and blame (gradient)
        error_scaled, loss,  loss_gradient = self.judge_pass(sample, prediction_raw)

        # Step 3: Backwards Pass -  Delegate to models logic for backprop or call default if not overridden
        self.back_pass(sample, loss_gradient)  # Call model-specific logic

        # 🎯 Record blame and weight updates or NeuroForge             (⬅️ Last step we need)
        self.VCR.record_blame_calculations  (self.blame_calculations)           # Write and clear error signal calculations to db for NeuroForge popup
        self.VCR.record_weight_updates      (self.weight_update_calculations, "update")   # Write and clear distribute error calculations to db for NeuroForge popup
        return error_scaled, loss, loss_gradient


    ################################################################################################
    ################################ SECTION 1 - Training Default Methods ##########################
    ################################################################################################
    def forward_pass(self, training_sample: Tuple[float, float, float]) -> None:
        """
        🚀 Computes forward pass for each neuron in the XOR MLP.
        🔍 Activation of Output neuron will be considered 'Raw Prediction'
        Args:
            training_sample: tuple where first elements are inputs and last element is target (assume one target)
        """

        input_values = training_sample[:-1]

        # 🚀 Compute raw sums + activations for each layer
        for layer_idx, layer in enumerate(Neuron.layers):  # Loop through all layers
            prev_activations = input_values if layer_idx == 0 else [n.activation_value for n in Neuron.layers[layer_idx - 1]]
            for neuron in layer:
                neuron.raw_sum = sum(input_val * weight for input_val, weight in zip(prev_activations, neuron.weights))
                neuron.raw_sum += neuron.bias
                neuron.activate()
                #print(f"neuron.activation={neuron.activation_value}")
        return  Neuron.layers[-1][0].activation_value  # Extract output neuron’s activation

    def back_pass(self, training_sample : Tuple[float, float, float], loss_gradient: float):
        """
        # Step 1: Compute blame for output neuron
        # Step 2: Compute blame for hidden neurons
        # Step 3: Adjust weights (Spread the blame)
        Args:
            training_sample (Tuple[float, float, float]): All inputs except the last which is the target
            loss_gradient (float) Derivative of the loss function with respect to the prediction.
        """

        # 🎯 Step 1: Compute blame (error signal) for output neuron
        self.back_pass__determine_blame_for_output_neuron(loss_gradient)

        # 🎯 Step 2: Compute blame (error signals) for hidden neurons        #    MUST go in reverse order AND MUST be based on weights BEFORE they are updated.(weight as it was during forward prop
        for layer_index in range(len(Neuron.layers) - 2, -1, -1):   # Exclude output layer
            for hidden_neuron in Neuron.layers[layer_index]:        # Iterate over current hidden layer
                self.back_pass__determine_blame_for_a_hidden_neuron(hidden_neuron)

        # 🎯 Step 3: Adjust weights for the output neuron
        self.back_pass__spread_the_blame(training_sample)        

    def back_pass__determine_blame_for_output_neuron(self, loss_gradient: float):
        """
        Calculate error_signal(gradient) for output neuron.
        Assumes one output neuron and that loss_gradient has already been calculated.
        Args:
            training_sample (Tuple[float, float, float]): All inputs except the last which is the target
            loss_gradient (float) Derivative of the loss function with respect to the prediction.
        """
        activation_gradient                 = Neuron.output_neuron.activation_gradient
        blame                               = loss_gradient * activation_gradient
        Neuron.output_neuron.error_signal   = blame

    def back_pass__determine_blame_for_a_hidden_neuron(self, neuron: Neuron):
        """
        Calculate the error signal for a hidden neuron by summing the contributions from all neurons in the next layer.
        args: neuron:  The neuron we are calculating the error for.
        """
        activation_gradient         = neuron.activation_gradient
        total_backprop_error        = 0  # Sum of (next neuron error * connecting weight)        
        
        # 🔄 Loop through each neuron in the next layer
        for next_neuron in Neuron.layers[neuron.layer_id + 1]:  # Next layer neurons
            weight_to_next          =  next_neuron.weights_before[neuron.position]  # Connection weight #TODO is weights before requried here?  I dont think so
            error_from_next         =  next_neuron.error_signal  # Next neuron’s error signal
            total_backprop_error    += weight_to_next * error_from_next  # Accumulate contributions
            neuron.error_signal     =  activation_gradient * total_backprop_error # 🔥 Compute final error signal for this hidden neuron

            # 🔹 Store calculation step as a structured tuple, now including weight index
            self.blame_calculations.append([
                self.epoch+1, self.iteration+1, self.config.gladiator_name, neuron.nid, next_neuron.position,
                weight_to_next, "*", error_from_next, "=", None, None, weight_to_next * error_from_next
            ])        

    def back_pass__spread_the_blame(self, training_sample : Tuple[float, float, float]):
        """
        Loops through all neurons, gathering the information required to update that neurons weights
        Args:
            training_sample (Tuple[float, float, float]): All inputs except the last which is the target
        """
        # Iterate backward through all layers, including the output layer
        for layer_index in range(len(Neuron.layers) - 1, -1, -1):  # Start from the output layer
            if layer_index == 0:                        # For the first layer (input layer), use raw inputs
                prev_layer = training_sample[:-1]       # Exclude the target
            else:                                       # For other layers, use activations from the previous layer
                prev_layer = [n.activation_value for n in Neuron.layers[layer_index - 1]]
            for neuron in Neuron.layers[layer_index]:   # Adjust weights for each neuron in the current layer
                self.back_pass__update_neurons_weights(neuron, prev_layer)

    def back_pass__update_neurons_weights(self, neuron: Neuron, prev_layer_values: list[float]) -> None:
        blame = neuron.error_signal
        input_vector = [1.0] + list(prev_layer_values)

        self.weight_update_calculations.extend( #Note list from _Finalize is gathered in VCR
            self.config.optimizer.update(
                neuron, input_vector, blame, self.total_iterations ,
                config      = self.config,
                epoch       = self.epoch + 1,
                iteration   = self.iteration + 1,
                batch_id    = self.VCR.batch_id
            )
        )
        self.total_iterations += len(input_vector)

    ############################### END OF BACKPASS ###############################
    ############################### END OF BACKPASS ###############################
    ############################### END OF BACKPASS ###############################


    def judge_pass(self, sample, prediction_raw: float):
        """
        Computes error, loss, and blame based on the configured loss function.
        """
        target  = sample[-1]
        error   = target - prediction_raw                                   # ✅ Simple error calculation
        loss    = self.config.loss_function(prediction_raw, target)         # ✅ Compute loss dynamically
        blame   = self.config.loss_function.grad(prediction_raw, target)    # ✅ Compute correction (NO inversion!)      #print(f"🔎 DEBUG: Target={target}, Prediction={prediction_raw}, Error={error}, Loss={blame}, Correction={blame}")
        return error, loss, blame                                           # ✅ Correction replaces "loss gradient"

    ################################################################################################
    ################################ SECTION 3 - Initialization ####################################
    ################################################################################################

    def snapshot_weights(self, from_suffix: str, to_suffix: str):
        """
        Copies weights and biases from one named attribute to another for all neurons.
        Example:
            snapshot_weights("", "_best")      # Save current as best
            snapshot_weights("_best", "")      # Restore best into active
        """
        for layer in Neuron.layers:
            for neuron in layer:
                from_weights = getattr(neuron, f"weights{from_suffix}")
                from_bias    = getattr(neuron, f"bias{from_suffix}")

                setattr(neuron, f"weights{to_suffix}", np.copy(from_weights))
                setattr(neuron, f"bias{to_suffix}", from_bias)


    def update_best_weights_if_new_lowest_error(self, current_error: float):
        """
        Checks if the current error is the lowest seen so far, and if so,
        stores weights and bias as the new best.
        """
        if not hasattr(self, 'lowest_error') or current_error < self.lowest_error:
            self.lowest_error = current_error
            self.snapshot_weights("", "_best")

    def restore_best_weights_from_run(self):
        """
        Restores the best weights and biases recorded during the run.
        Assumes snapshot_weights("_best", "") was used to store them.
        """
        if not hasattr(self, 'lowest_error'):
            print("⚠️ No best weights found — 'lowest_error' was never set.")
            return

    def get_flat_initializers(self, architecture: List[int], initializers: List[ScalerWeightInitializer]) -> List[ScalerWeightInitializer]:
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

    def initialize_neurons(self,  architecture: List[int] , initializers: List[ScalerWeightInitializer] = None, hidden_activation: StrategyActivationFunction = None, output_activation: StrategyActivationFunction = None):
        """
        Initializes neurons based on the specified architecture, using appropriate weight initializers.

        Args:
            architecture (List[int]): Number of neurons per hidden layer and output layer
            initializers (List[WeightInitializer]): A list of weight initializers.
            hidden_activation (ActivationFunction): The activation function for hidden layers.
        """

        if architecture is None:
             raise ValueError("Blueprint is empty - this shouldn't be possible - defect alert")

        # Ensure initializer list matches neuron count
        flat_initializers = self.get_flat_initializers(architecture, initializers)
        #print(f"Checking flat_initializers: {flat_initializers}")

        input_count             = self.training_data.input_count
        hidden_activation       = hidden_activation or self.config.hidden_activation
        output_activation       = output_activation or self.config.loss_function.recommended_output_activation #None indicates no restriction
        nid                     = -1

        Neuron.neurons.clear()
        Neuron.layers.clear()
        #print(f"architecture1 = {self.config.architecture}  Neuron count = {len(Neuron.neurons)}")
        for layer_index, layer_size in enumerate(architecture):
            #print(f"Index: {layer_index}, Layer size: {layer_size}")

            if layer_index == 0:    # For first layer of neurons
                num_of_weights = self.training_data.input_count
            else:
                num_of_weights = architecture[layer_index - 1]

            for neuron_index in range(layer_size):
                nid += 1
                activation = output_activation if layer_index==len(architecture)-1 else hidden_activation
                #print(f"Creating Neuron {nid}  in layer{layer_index}  len(architecture)={len(architecture)} - Act = {activation.name}")
                neuron = Neuron(
                    nid                 = nid,
                    num_of_weights      = num_of_weights,
                    learning_rate       = self.config.learning_rate,
                    weight_initializer  = flat_initializers[nid],  # Assign correct initializer
                    layer_id            = layer_index,
                    activation          = activation
                )

        #print(f"architecture2 = {self.config.architecture}  Neuron count = {len(Neuron.neurons)}")

    ################################################################################################
    ################################ SECTION 4 - Binary Decision logic ####################################
    ################################################################################################
    @property
    def bd_class_alpha(self):
        return self._bd_class_alpha

    @bd_class_alpha.setter
    def bd_class_alpha(self, value):
        rule = self.config.loss_function.bd_rules[2]  # Extract the modification rule
        if rule.startswith("Error"):
            raise ValueError(f"🚨 Modification of bd_class_alpha is not allowed for this loss function! {rule}")
        if rule.startswith("Warning"):
            print(f"⚠️ {rule}")  # Show warning but allow modification
        self._bd_class_alpha = value

    @property
    def bd_class_beta(self):
        return self._bd_class_beta

    @bd_class_beta.setter
    def bd_class_beta(self, value):
        rule = self.config.loss_function.bd_rules[2]  # Extract the modification rule
        if rule.startswith("Error"):
            raise ValueError(f"🚨 Modification of bd_class_alpha is not allowed for this loss function! {rule}")
        if rule.startswith("Warning"):
            print(f"⚠️ {rule}")  # Show warning but allow modification
        self._bd_class_beta = value

    @property
    def bd_threshold(self):
        return self._bd_threshold

    @bd_threshold.setter
    def bd_threshold(self, value):
        rule = self.config.loss_function.bd_rules[3]  # Extract the modification rule
        if rule.startswith("Error"):
            raise ValueError(f"🚨 Modification of Threshold is not allowed for this loss function! {rule}")
        if rule.startswith("Warning"):
            print(f"⚠️ {rule}")  # Show warning but allow modification
        self._bd_threshold = value


    @property
    def last_epoch_mae(self):
        return self.total_error_for_epoch/self.config.training_data.sample_count

    @property
    def weights(self):
        """
        Getter for learning rate.
        """
        return Neuron.neurons[0].weights
    #@property
    def learning_rateNOT_HERE_IN_CONFIG(self):
        """
        Getter for learning rate.
        """
        return self._learning_rate

    #@learning_rate.setter
    def learning_rateNOT_HERE_IN_CONFIG(self, new_learning_rate: float):
        """
        Updates the learning rate for the Gladiator and ensures all neurons reflect the change.

        Args:
            new_learning_rate (float): The new learning rate to set.
        """

        self._learning_rate = new_learning_rate
        self.config.learning_rate = new_learning_rate
        for neuron in Neuron.neurons:
            neuron.set_learning_rate(new_learning_rate)
            #if neuron.nid == 0:
                #self.db.query_print("Select 1 as in_LR_Setter")
                #print(f"self._learning_rate={self._learning_rate}")
            #neuron.learning_rate_report("Before")
            #if neuron.nid == 0:
                #neuron.set_learning_rate(new_learning_rate)
                #neuron.learning_rate_report("After")

    def print_reproducibility_info(self, epoch_count):
        print("\n🧬 Reproducibility Snapshot")
        print("────────────────────────────")
        print(f"Arena:             {self.config.training_data.arena_name}")
        print(f"Gladiator:         {self.config.gladiator_name}")
        print(f"Architecture:      {self.config.architecture}")
        print(f"Problem Type:      {self.config.training_data.problem_type}")
        print(f"Loss Function:     {self.config.loss_function.__class__.__name__}")
        print(f"Hidden AF:         {self.config.hidden_activation.name}")
        print(f"Output AF:         {self.config.output_activation.name}")
        print(f"Weight Init:       {self.config.initializer.name}")
        print(f"Data Norm Scheme:  {self.config.training_data.norm_scheme}")
        print(f"Seed:              {self.config.hyper.random_seed}")
        print(f"Learning Rate:     {self._learning_rate}")
        print(f"Epochs Run:        {epoch_count}")
        print(f"Convergence Rule:  {self.config.cvg_condition}")
        #print(f"Final Error:       {self.config.db.get_final_mae(self.config.gladiator_name):.4f}")
        #print(f"Final Accuracy:    {self.config.db.get_final_accuracy(self.config.gladiator_name):.2%}")
        #print(f"Runtime (secs):    {self.config.seconds:.2f}")


        print("────────────────────────────\n")
        def on_epoch_end(self, epoch, error_summary):
            """
            Optional override: Called after each epoch.
            """
            pass

        def on_training_complete(self):
            """
            Optional override: Called after training run completes.
            """
            pass
