from abc import ABC
from json import dumps
from src.Legos.ActivationFunctions import *
from src.Legos.Optimizers import Optimizer_Adam
from src.engine.MgrSQL import MgrSQL
from src.engine.Config import Config
from src.engine.Neuron import Neuron
from datetime import datetime


from src.engine.Utils_DataClasses import Iteration, ReproducibilitySnapshot
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
    def __init__(self,  config: Config):
        self.gladiator          = config.gladiator_name
        self.db                 = config.db
        self.hyper              = config.hyper
        self.training_data      = config.training_data              # Only needed for sqlMgr ==> self.ramDb = args[3]
        self.neurons            = []
        #self.layers             = []                        # Layered structure
        self.neuron_count       = 0                         # Default value
        self.total_iterations   = 1                         # Timestep for optimizers such as adam
        self.training_samples   = None                      # To early to get, becaus normalization wouldn't be applied yet self.training_data.get_list()   # Store the list version of training data
        self.mgr_sql            = MgrSQL(config, self.gladiator, self.hyper, self.training_data, self.neurons, config.db) # Args3, is ramdb
        self._learning_rate     = self.hyper.default_learning_rate #todo set this to all neurons learning rate
        self.number_of_epochs   = self.hyper.epochs_to_run
        self._full_architecture = None
        self._bd_threshold       = None
        self._bd_class_alpha     = None
        self._bd_class_beta      = None
        self.total_error_for_epoch          = 0
        self.iteration          = 0
        self.epoch              = 0
        self.too_high_adjst     = self.training_data.input_max * 5 #TODO make 5 hyperparamter
        self.max_adj            = self.training_data.everything_max_magnitude()
        self.config             = config
        self.blame_calculations = []
        self.weight_update_calculations = []
        self.convergence_phase  = "watch"
        self.retrieve_setup_from_model()


    ################################################################################################
    ################################ SECTION 1 - pipeline ####################################
    ################################################################################################
    def setup_backprop_headers(self): # MUST OCCUR AFTER CONFIGURE MODEL SO THE OPTIMIZER IS SET
        if self.config.optimizer.backprop_popup_headers is not None:
            self.config.backprop_headers = self.config.optimizer.backprop_popup_headers


    def retrieve_setup_from_model(self):
        self.configure_model(self.config)  #Typically overwritten in base class.
        self.setup_backprop_headers()
        self.initialize_neurons(
            architecture=self.config.architecture.copy(),  # Avoid mutation
            initializers=[self.config.initializer],  # <- List of 1 initializer
            hidden_activation=self.config.hidden_activation,
            output_activation=self.config.output_activation
            or self.config.loss_function.recommended_output_activation)

        self.customize_neurons(self.config)

    def configure_model(self, config: Config):
        pass
    def customize_neurons(self,config: Config):
        pass

    def train(self) -> tuple[str, list[int]]:
        """
        Main method invoked from Framework to train model.

        Returns:
            tuple[str, list[int]]: A tuple containing:
                - converged_signal (str): Which convergence signal(s) triggered early stop.
                - full_architecture (list[int]): Architecture of the model including hidden and output layers.
        """

        self.config.loss_function.validate_activation_functions()
        if self.neuron_count == 0:
            self.initialize_neurons([]) #Defaults to 1 when it adds the output
        self.check_binary_decision_info()
        self.training_samples = self.training_data.get_list()           # Store the list version of training data

        for epoch in range(self.number_of_epochs):                      # Loop to run specified # of epochs
            convg_signal= self.run_an_epoch(epoch)                                # Call function to run single epoch
            if convg_signal !="":                                 # Converged so end early
                if convg_signal == "fix_temporarilydisabled":
                    self.convergence_phase = "fix"
                else:
                    snapshot = ReproducibilitySnapshot.from_config(self._learning_rate, epoch, self.last_epoch_mae, self.config)
                    return convg_signal, self._full_architecture, snapshot
        snapshot = ReproducibilitySnapshot.from_config(self._learning_rate, epoch, self.last_epoch_mae, self.config)
        return "Did not converge", self._full_architecture, snapshot       # When it does not converge still return info

    def validate_output_activation_functionDeleteME(self):
        """
        Check if loss function requires specific activation function in output neuron.
        """
        # 🚨 Validate activation function before training begins
        allowed_activations = self.config.loss_function.allowed_activation_functions
        actual_activation = Neuron.output_neuron.activation

        if allowed_activations is not None and actual_activation not in allowed_activations:
            raise ValueError(
                f"🚨 {actual_activation} is not compatible with Loss function {self.config.loss_function.name}. "
                f"\nAllowed: {', '.join([act.name for act in allowed_activations])}"
            )

    def run_an_epoch(self, epoch_num: int) -> str:
        """
        Executes a training epoch i.e. trains on all samples

        Args:
            epoch_num (int) : number of epoch being executed
        Returns:
            convergence_signal (str) : If not converged, empty string, otherwise signal that detected convergence
        """

        self.epoch = epoch_num      # Set so the child model has access
        if epoch_num % 100 == 0 and epoch_num!=0:
                print (f"Epoch: {epoch_num} for {self.gladiator} Loss = {self.last_epoch_mae} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.total_error_for_epoch = 0
        for i, sample in enumerate(self.training_samples):  # Loop through all training data
            #print (f"self.total_error_for_epoch={self.total_error_for_epoch}\tself.last_epoch_mae={self.last_epoch_mae}")
            self.iteration = i      # Set so the model has access
            sample = np.array(sample)  # Convert sample to NumPy array
            inputs = sample[:-1]
            target = sample[-1]
            self.snapshot_weights("", "_before")
            error, loss,  loss_gradient = self.optimizer_simplified_descent(sample, inputs, target)
            prediction_raw = Neuron.layers[-1][0].activation_value  # Extract single neuron’s activation
            self.total_error_for_epoch += abs(error)
            #if error < self.config.lowest_error:    # New lowest error
            #    self.config.lowest_error = error
            #    self.config.lowest_error_epoch = epoch_num

            #If binary decision apply step logic.
            prediction = prediction_raw # Assume regression
            if self.training_data.problem_type == "Binary Decision":
                prediction = self.bd_class_beta if prediction_raw >= self.bd_threshold else self.bd_class_alpha
                #print(f"AFTER\tself.bd_class_beta={self.bd_class_beta}\tprediction={prediction}\tself.bd_threshold={self.bd_threshold} self.bd_class_alpha=\t{ self.bd_class_alpha}")

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
                loss_function=self.config.loss_function.name,
                loss_gradient=loss_gradient,
                accuracy_threshold=self.hyper.accuracy_threshold,
            )
            self.mgr_sql.record_iteration(iteration_data, Neuron.layers)
            # I NEED TO TEST THIS WHEN I NEED IT.self.update_best_weights_if_new_lowest_error(self.last_epoch_mae)
        return self.mgr_sql.finish_epoch(epoch_num + 1)      # Finish epoch and return convergence signal

    def optimizer_simplified_descent(self, sample, inputs, target):
        # Step 1: Forward pass
        self.forward_pass(sample)  # Call model-specific logic
        prediction_raw = Neuron.layers[-1][0].activation_value  # Extract single neuron’s activation

        # Step 3: Delegate to models logic for Validate_pass :)
        error, loss,  loss_gradient = self.validate_pass(target, prediction_raw )
        #loss_gradient = self.watch_for_explosion(loss_gradient)

        # Step 4: Delegate to models logic for backporop.
        self.back_pass(sample, loss_gradient)  # Call model-specific logic
        
        # 🎯 Record what was done for NeuroForge             (⬅️ Last step we need)
        self.record_blame_calculations()                    # Write error signal calculations to db for NeuroForge popup
        self.record_weight_updates()                        # Write distribute error calculations to db for NeuroForge popup
        return error, loss,  loss_gradient

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
                self.epoch+1, self.iteration+1, self.gladiator, neuron.nid, next_neuron.position,
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


        self.weight_update_calculations.extend(
            self.config.optimizer.update(
                neuron, input_vector, blame, self.total_iterations ,
                config=self.config,
                epoch=self.epoch + 1,
                iteration=self.iteration + 1,
                gladiator=self.gladiator
            )
        )
        self.total_iterations += len(input_vector)



    def back_pass__update_neurons_weightsv1(self, neuron: Neuron, prev_layer_values: list[float]) -> None:
        blame = neuron.error_signal
        input_vector = [1.0] + list(prev_layer_values)
        t = 0 # self.epoch * self.total_iterations + self.iteration + 1

        adjustments = self.config.optimizer.update(neuron, input_vector, blame, t, self.config)

        for i, adj in enumerate(adjustments):
            self.weight_update_calculations.append([
                self.epoch + 1, self.iteration + 1, self.gladiator, neuron.nid, i,
                input_vector[i], "*", blame, "*", neuron.learning_rates[i], "=", adj
            ])

        """  
            # Optional: Capture the actual adjustment if desired
            adjustment = grad * neuron.learning_rates[i]  # Only accurate for SGD
            self.weight_update_calculations.append([
                self.epoch + 1, self.iteration + 1, self.gladiator, neuron.nid, i,
                prev_value, "*", blame, "*", neuron.learning_rates[i], "=", adjustment
            ])
            """

    def back_pass__update_neurons_weights_NoOptimizer(self, neuron: Neuron, prev_layer_values: list[float]) -> None:
        """
        Updates weights for a neuron based on blame (error signal).
        Args:
            neuron: The neuron that will have its weights updated to.
            prev_layer_values: (list[float]) Activations from the previous layer or inputs for first hidden layer
        """
        blame = neuron.error_signal                             # Get the culpability assigned to this neuron
        input_vector = [1.0] + list(prev_layer_values)

        for i, prev_value in enumerate(input_vector):
            learning_rate = neuron.learning_rates[i]
            adjustment = prev_value * blame * learning_rate

            # 🔹 Update
            if i == 0:
                neuron.bias -= adjustment
            else:
                neuron.weights[i - 1] -= adjustment

            # 🔹 Store structured calculation for weights
            self.weight_update_calculations.append([
                self.epoch + 1, self.iteration + 1, self.gladiator, neuron.nid, i,
                prev_value, "*", blame, "*", learning_rate, "=", adjustment
            ])


    #################################################
    #################################################
    #################################################
    #################################################
    def back_pass__distribute_errorAdaptive(self, neuron: Neuron, prev_layer_values: list[float]) -> None:
        """
        Updates weights for a neuron based on blame (error signal).
        args: neuron: The neuron that will have its weights updated to.

        - First hidden layer uses inputs from training data.
        - All other neurons use activations from the previous layer.
        """
        error_signal = neuron.error_signal

        for i, (w, prev_value) in enumerate(zip(neuron.weights, prev_layer_values)):
            weight_before = neuron.weights[i]
            adjustment  = prev_value * error_signal *  neuron.learning_rates[i+1] #1 accounts for bias in 0  #So stupid to go down hill they look uphill and go opposite
            if abs(adjustment) > self.too_high_adjst: #Explosion detection
                adjustment = 0
                neuron.learning_rates[i+1] *= 0.5     #reduce neurons LR
            # **💡 Growth Factor: Gradually Increase LR if too slow**

            #elif not is_exploding(weight) and not is_oscillating(weight):
            else:
                neuron.learning_rates[i] *= 1.05  # Boost LR slightly if it looks stable

            neuron.weights[i] -= adjustment
            #print(f"trying to find path down{self.epoch+1}, {self.iteration+1}\tprev_value{prev_value}\terror_signal{error_signal}\tlearning_rate{learning_rate}\tprev_value{adjustment}\t")

            # 🔹 Store structured calculation for weights
            self.weight_update_calculations.append([
                # epoch, iteration, model_id, neuron_id, weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result
                self.epoch+1, self.iteration+1, self.gladiator, neuron.nid, i+1,
                prev_value, "*", error_signal, "*", neuron.learning_rates[i+1], "=", adjustment
            ])


        # Bias update
        adjustment_bias = neuron.learning_rates[0] * error_signal
        if abs(adjustment_bias) > self.too_high_adjst: #Explosion detection
            adjustment_bias = 0
            neuron.learning_rates[0] *= 0.5     #reduce neurons LR
        else:
            neuron.learning_rates[0] *= 1.05     #reduce neurons LR
        neuron.bias -= adjustment_bias

        # 🔹 Store structured calculation for bias
        self.weight_update_calculations.append([
        # epoch, iteration, model_id, neuron_id, weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result
            self.epoch+1 , self.iteration+1, self.gladiator, neuron.nid, 0,
                "1", "*", error_signal, "*", neuron.learning_rates[0],   "=", adjustment_bias
            ])


    def convert_numpy_scalars_because_python_is_weak(self, row):
        """
        Converts any NumPy scalar values in the given row to their native Python types.
        Friggen ridiculous it was converting either 0 to null or 1 to 0.... what a joke this language is
        """
        return [x.item() if hasattr(x, 'item') else x for x in row]

    def record_weight_updates(self):
        """
        Inserts all weight update calculations for the current iteration into the database.
        """

        if self.config.optimizer == Optimizer_Adam:
            sql = """
                INSERT INTO DistributeErrorCalcs 
                (epoch, iteration, model_id, nid, weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        else:
            sql = """
                INSERT INTO DistributeErrorCalcs 
                (epoch, iteration, model_id, nid, weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result)
                VALUES (?, ?, ?, ?, ?, CAST(? AS REAL), ?, CAST(? AS REAL), ?, CAST(? AS REAL), ?, CAST(? AS REAL))
            """

        #print("About to insert")
        #for row in self.weight_update_calculations:
        #    print(row)

        # Convert each row to ensure any numpy scalars are native Python types
        converted_rows = [self.convert_numpy_scalars_because_python_is_weak(row) for row in self.weight_update_calculations]
        #print(f"converted rows = {converted_rows}")
        self.db.executemany(sql, converted_rows)
        self.weight_update_calculations.clear()
        #self.db.query_print("SELECT * FROM DistributeErrorCalcs WHERE iteration = 2 and nid = 0 ORDER BY weight_index")


    def record_blame_calculations(self):
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
        converted_rows = [self.convert_numpy_scalars_because_python_is_weak(row) for row in self.blame_calculations]
        #print(f"BLAME {self.blame_calculations}")

        #Heads up, sometimes overflow error look like key violation here
        self.db.executemany(sql, self.blame_calculations)
        self.blame_calculations.clear()

    def validate_pass(self, target: float, prediction_raw: float):
        """
        Computes error, loss, and blame based on the configured loss function.
        """
        error   = target - prediction_raw  # ✅ Simple error calculation
        loss    = self.config.loss_function(prediction_raw, target)  # ✅ Compute loss dynamically
        blame   = self.config.loss_function.grad(prediction_raw, target)  # ✅ Compute correction (NO inversion!)      #print(f"🔎 DEBUG: Target={target}, Prediction={prediction_raw}, Error={error}, Loss={blame}, Correction={blame}")
        return error, loss,  blame  # ✅ Correction replaces "loss gradient"

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

    def initialize_neurons(self,  architecture: List[int] , initializers: List[WeightInitializer] = None, hidden_activation: ActivationFunction = None, output_activation: ActivationFunction = None):
        """
        Initializes neurons based on the specified architecture, using appropriate weight initializers.

        Args:
            architecture (List[int]): Number of neurons per hidden layer.
            initializers (List[WeightInitializer]): A list of weight initializers.
            hidden_activation (ActivationFunction): The activation function for hidden layers.
        """
        print("Initializing!!!!!!!!!!!!!!!!!")
        Neuron.reset_layers()
        if architecture is None:
            architecture = []  # Default to no hidden layers
        architecture.append(1)  # Add output neuron

        # Ensure initializer list matches neuron count
        flat_initializers = self.get_flat_initializers(architecture, initializers)
        print(f"Checking flat_initializers: {flat_initializers}")

        input_count             = self.training_data.input_count
        hidden_activation       = hidden_activation or self.config.hidden_activation
        output_activation       = output_activation or self.config.loss_function.recommended_output_activation #None indicates no restriction
        self._full_architecture = [input_count] + architecture  # Store the full architecture
        nid                     = -1
        self.neurons.clear()
        Neuron.layers.clear()

        for layer_index, neuron_count in enumerate(architecture):
            num_of_weights = self._full_architecture[layer_index]
            for neuron_index in range(neuron_count):
                nid += 1

                activation = output_activation if layer_index==len(architecture)-1 else hidden_activation
                #print(f"Creating Neuron {nid}  in layer{layer_index}  len(architecture)={len(architecture)} - Act = {activation.name}")
                neuron = Neuron(
                    nid                 = nid,
                    num_of_weights      = num_of_weights,
                    learning_rate       = self.learning_rate,
                    weight_initializer  = flat_initializers[nid],  # Assign correct initializer
                    layer_id            = layer_index,
                    activation          = activation
                )
                self.neurons.append(neuron)
        self.neuron_count = len(self.neurons)


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

    def check_binary_decision_info(self):
        print (f"self.config.loss_function={self.config.loss_function}")
        if self.training_data.problem_type == "Binary Decision":
            a, b, c = self.training_data.apply_binary_decision_targets_for_specific_loss_function(self.config.loss_function)
            # Only update if still None (i.e., not set by the Gladiator)
            if self._bd_class_alpha is None:
                self._bd_class_alpha = a  # Directly setting avoids triggering warnings
            if self._bd_class_beta is None:
                self._bd_class_beta = b
            if self._bd_threshold is None:
                self._bd_threshold = c

    @property
    def last_epoch_mae(self):
        return self.total_error_for_epoch/self.config.training_data.sample_count

    @property
    def weights(self):
        """
        Getter for learning rate.
        """
        return self.neurons[0].weights
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
        for neuron in Neuron.neurons:
            neuron.set_learning_rate(new_learning_rate)

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
