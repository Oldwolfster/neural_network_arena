from abc import ABC, abstractmethod
from json import dumps
from src.engine.ActivationFunction import *
from src.engine.MgrSQL import MgrSQL
from src.engine.ModelConfig import ModelConfig
from src.engine.Neuron import Neuron
from datetime import datetime


from src.engine.Utils_DataClasses import Iteration
from src.engine.WeightInitializer import *
from typing import List

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
    def __init__(self,  config: ModelConfig):
        self.gladiator          = config.gladiator_name
        self.db                 = config.db
        self.hyper              = config.hyper
        self.training_data      = config.training_data              # Only needed for sqlMgr ==> self.ramDb = args[3]
        self.neurons            = []
        #self.layers             = []                        # Layered structure
        self.neuron_count       = 0                         # Default value
        self.training_data      . reset_to_default()
        self.training_samples   = None                      # To early to get, becaus normalization wouldn't be applied yet self.training_data.get_list()   # Store the list version of training data
        self.mgr_sql            = MgrSQL(self.gladiator, self.hyper, self.training_data, self.neurons, config.db) # Args3, is ramdb
        self._learning_rate     = self.hyper.default_learning_rate #todo set this to all neurons learning rate
        self.number_of_epochs   = self.hyper.epochs_to_run
        self._full_architecture  = None
        self.last_lost          = 0
        self.iteration          = 0
        self.epoch              = 0
        self.config             = config
        self._simpletron(config)        #Check if support needed for old optimzer
        self.error_signal_calcs = []
        self.distribute_error_calcs = []

    ################################################################################################
    ################################ SECTION 1 - Simpletron Support ##########################
    ################################################################################################
    def _simpletron(self, config:ModelConfig):
        if hasattr(self, 'training_iteration'):
            config.optimizer="simpletron"

    ################################################################################################
    ################################ SECTION 1 - pipeline ####################################
    ################################################################################################

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
        print(f"epoch={self.epoch}")
        #print(f"\tepoch\titeration\tinput\ttarget\tprediction\terror\tweight before adj\tfinal weight")
        if epoch_num % 100 == 0 and epoch_num!=0:
                print (f"Epoch: {epoch_num} for {self.gladiator} Loss = {self.last_lost} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for i, sample in enumerate(self.training_samples):  # Loop through all training data
            self.iteration = i      # Set so the model has access
            sample = np.array(sample)  # Convert sample to NumPy array
            inputs = sample[:-1]
            target = sample[-1]
            self.snapshot_weights_as_weights_before()
            prediction_raw = 0
            if self.config.optimizer=="simplified_descent":
                error, loss,  loss_gradient = self.optimizer_simplified_descent(sample, inputs, target)
                prediction_raw = Neuron.layers[-1][0].activation_value  # Extract single neuron’s activation
            elif self.config.optimizer=="simpletron":
                error, loss,  loss_gradient, prediction_raw = self.optimizer_simpletron(sample, inputs,target)
            else:
                raise TypeError("Optimizer not supported")
            #print(f"prediction_raw={prediction_raw}\ttarget={target}\terror={error}")
            self.last_lost = loss

            #If binary decision apply step logic.
            prediction = prediction_raw # Assyme regression
            if self.training_data.problem_type == "Binary Decision":
                prediction = 1 if prediction_raw >= 0.5 else 0

            #print(f"{self.gladiator}\t{self.epoch}\t{self.iteration}\t{inputs[0]}\t{target}\t{prediction}\t{error}\t{self.weights[0]}\t{self.neurons[0].weights_before[0]}\t{self.training_data.problem_type}")

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
        return self.mgr_sql.finish_epoch()      # Finish epoch and return convergence signal
    def optimizer_simpletron(self, sample, inputs, target):
        prediction_raw = self.training_iteration(sample)
        error = target - prediction_raw
        loss = error
        loss_gradient = error
        return error, loss,  loss_gradient, prediction_raw
    def optimizer_simplified_descent(self, sample, inputs, target):
        # Step 1: Forward pass
        self.forward_pass(sample)  # Call model-specific logic
        prediction_raw = Neuron.layers[-1][0].activation_value  # Extract single neuron’s activation

        # Step 3: Delegate to models logic for Validate_pass :)
        error, loss,  loss_gradient = self.validate_pass(target, prediction_raw )
        loss_gradient = self.watch_for_explosion(loss_gradient)
        # Step 4: Delegate to models logic for backporop.
        self.back_pass(sample, loss_gradient)  # Call model-specific logic
        return error, loss,  loss_gradient



    def watch_for_explosion(self,correction: float) ->float:        # 🚨 Detect gradient explosion
        if abs(correction) > self.config.hyper.gradient_clip_threshold:
            print(f"🚨 Gradient Explosion Detected! Clipping correction: {correction}")
            correction = np.sign(correction) * self.config.hyper.gradient_clip_threshold  # Clip to max allowed
        return  correction

    ################################################################################################
    ################################ SECTION 1 - Training Default Methods ##########################
    ################################################################################################
    def forward_pass(self, training_sample):
        """
        Computes forward pass for each neuron in the XOR MLP.
        """
        #print("🚀Using Default Forward pass - to customize override forward_pass")
        input_values = training_sample[:-1]

        # 🚀 Compute raw sums + activations for each layer
        for layer_idx, layer in enumerate(Neuron.layers):  # Exclude output layer
            prev_activations = input_values if layer_idx == 0 else [n.activation_value for n in Neuron.layers[layer_idx - 1]]

            for neuron in layer:
                neuron.raw_sum = sum(input_val * weight for input_val, weight in zip(prev_activations, neuron.weights))
                neuron.raw_sum += neuron.bias
                neuron.activate()


    def back_pass(self, training_sample, loss_gradient: float):
        #print("🚀Using Default back_pass - to customize override back_pass")
        output_neuron = Neuron.layers[-1][0]

        # Step 1: Compute error signal for output neuron
        self.back_pass__error_signal_for_output(loss_gradient)

        # Step 2: Compute error signals for hidden neurons
        # * MUST go in reverse order!
        # * MUST be based on weights BEFORE they are updated.(weight as it was during forward prop
        for layer_index in range(len(Neuron.layers) - 2, -1, -1):  # Exclude output layer
            for hidden_neuron in Neuron.layers[layer_index]:  # Iterate over current hidden layer
                self.back_pass__error_signal_for_hidden(hidden_neuron)

        # Step 3: Adjust weights for the output neuron
        self.adjust_weights(training_sample)

        # Step 4: Adjust weights for the hidden neurons (⬅️ Last step we need)
        self.insert_error_signal_calcs()            # Write error signal calculations to db for NeuroForge popup
        self.insert_distribute_error_calcs()        # Write distribute error calculations to db for NeuroForge popup

    def back_pass__error_signal_for_output(self, loss_gradient: float):
        """
        Calculate error_signal(gradient) for output neuron.
        Assumes one output neuron and that loss_gradient has already been calculated.
        """
        output_neuron               = Neuron.layers[-1][0]
        activation_gradient         = output_neuron.activation_gradient
        error_signal                = loss_gradient * activation_gradient
        output_neuron.error_signal  = error_signal

    def adjust_weights(self, training_sample):
        """
        Adjust weights for all neurons in the network using backpropagation.
        This works for both perceptrons and MLPs.
        """
        # Iterate backward through all layers, including the output layer
        for layer_index in range(len(Neuron.layers) - 1, -1, -1):  # Start from the output layer
            if layer_index == 0:
                # For the first layer (input layer), use raw inputs
                prev_layer_activations = training_sample[:-1]  # Exclude the label
            else:
                # For other layers, use activations from the previous layer
                prev_layer_activations = [n.activation_value for n in Neuron.layers[layer_index - 1]]

            # Adjust weights for each neuron in the current layer
            for neuron in Neuron.layers[layer_index]:
                self.back_pass__distribute_error(neuron, prev_layer_activations)


    def back_pass__distribute_error(self, neuron: Neuron, prev_layer_values):
        """
        Updates weights for a neuron based on error signal.
        args: neuron: The neuron that will have its weights updated to.

        - First hidden layer uses inputs from training data.
        - All other neurons use activations from the previous layer.
        """
        learning_rate = neuron.learning_rate
        error_signal = neuron.error_signal
        weight_formulas = []
        #if neuron.nid    == 2 and self.epoch==1 and self.iteration<3:
        #print(f"WEIGHT UPDATE FOR epoch:{self.epoch}\tItertion{self.iteration}")

        for i, (w, prev_value) in enumerate(zip(neuron.weights, prev_layer_values)):
            weight_before = neuron.weights[i]
            #If calculating gradient tradional way (errpr *-2) then below shuold subtract not add. but it dont work
            #adjustment  = learning_rate * error_signal * prev_value #So stupid to go down hill they look uphill and go opposite
            adjustment  = prev_value * error_signal *  learning_rate  #So stupid to go down hill they look uphill and go opposite
            neuron.weights[i] += adjustment
            #print(f"{self.epoch+1}, {self.iteration+1}\tprev_value{prev_value}\terror_signal{error_signal}\tlearning_rate{learning_rate}\tprev_value{adjustment}\t")

            # 🔹 Store structured calculation for weights
            self.distribute_error_calcs.append([
                # epoch, iteration, model_id, neuron_id, weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result
                self.epoch+1, self.iteration+1, self.gladiator, neuron.nid, i+1,
                prev_value, "*", error_signal, "*", learning_rate, "=", adjustment
            ])

        if self.iteration+1==2:
            print("Mistake here!")
            for row in self.distribute_error_calcs:
                print(row)



        #if neuron.nid    == 2 and self.epoch==0 and self.iteration==0:
        #    print (f"All weights for neuron #{neuron.nid} epoch:  {self.epoch}\tItertion{self.iteration}\tWeights before=>{neuron.weights_before}\t THEY SHOULD BE -0.24442546 -0.704763 #Weights before=>{neuron.weights}")#seed 547298 LR = 1.0
        # Bias update
        adjustment_bias = learning_rate * error_signal
        neuron.bias += adjustment_bias

        # 🔹 Store structured calculation for bias

        self.distribute_error_calcs.append([
        # epoch, iteration, model_id, neuron_id, weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result
            self.epoch+1 , self.iteration+1, self.gladiator, neuron.nid, 0,
                "1", "*", error_signal, "*", learning_rate,   "=", adjustment_bias
            ])
    def convert_numpy_scalars_because_python_is_weak(self, row):
        """
        Converts any NumPy scalar values in the given row to their native Python types.
        Friggen ridiculous it was converting either 0 to null or 1 to 0.... what a joke this language is
        """
        return [x.item() if hasattr(x, 'item') else x for x in row]

    def insert_distribute_error_calcs(self):
        """
        Inserts all weight update calculations for the current iteration into the database.
        """
        sql = """
        INSERT INTO DistributeErrorCalcs 
        (epoch, iteration, model_id, nid, weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result)
        VALUES (?, ?, ?, ?, ?, CAST(? AS REAL), ?, CAST(? AS REAL), ?, CAST(? AS REAL), ?, CAST(? AS REAL))"""

        #print("About to insert")
        #for row in self.distribute_error_calcs:
        #    print(row)

        # Convert each row to ensure any numpy scalars are native Python types
        converted_rows = [self.convert_numpy_scalars_because_python_is_weak(row) for row in self.distribute_error_calcs]
        self.db.executemany(sql, converted_rows)
        self.distribute_error_calcs.clear()
        #self.db.query_print("SELECT * FROM DistributeErrorCalcs WHERE iteration = 2 and nid = 0 ORDER BY weight_index")

    def back_pass__error_signal_for_hidden(self, neuron: Neuron):
        """
        Calculate the error signal for a hidden neuron by summing the contributions from all neurons in the next layer.
        args: neuron:  The neuron we are calculating the error for.
        """
        activation_gradient = neuron.activation_gradient
        total_backprop_error = 0  # Sum of (next neuron error * connecting weight)
        neuron.error_signal_calcs=""

        #print(f"Calculating error signal epoch/iter:{self.epoch}/{self.iteration} for neuron {to_neuron.layer_id},{to_neuron.position}")
        # 🔄 Loop through each neuron in the next layer

        memory_efficent_way_to_store_calcs = []
        for next_neuron in Neuron.layers[neuron.layer_id + 1]:  # Next layer neurons
            #print (f"
            # getting weight and error from {to_neuron.layer_id},{to_neuron.position}")
            weight_to_next = next_neuron.weights_before[neuron.position]  # Connection weight #TODO is weights before requried here?  I dont think so
            error_from_next = next_neuron.error_signal  # Next neuron’s error signal
            total_backprop_error += weight_to_next * error_from_next  # Accumulate contributions
            # 🔹 Store calculation step as a structured tuple, now including weight index
            self.error_signal_calcs.append([
                self.epoch+1, self.iteration+1, self.gladiator, neuron.nid,
                next_neuron.position,  # Adding weight index for uniqueness
                weight_to_next, "*", error_from_next, "=", None, None, weight_to_next * error_from_next
            ])
            memory_efficent_way_to_store_calcs.append(f"{smart_format(weight_to_next)}!{smart_format(error_from_next)}@")
        neuron.error_signal_calcs = ''.join(memory_efficent_way_to_store_calcs)  # Join once instead of multiple string concatenations
        """
        print("**********************")
        print("**********************")
        print("**********************")
        for row in self.error_signal_calcs:
            print(row)
        print("**********************")
        print(neuron.error_signal_calcs)
        for row in memory_efficent_way_to_store_calcs:
            print(row)
        # 🔥 Compute final error signal for this hidden neuron
        """
        neuron.error_signal = activation_gradient * total_backprop_error

    def insert_error_signal_calcs(self):
        """
        Inserts all backprop calculations for the current iteration into the database.
        """
        #print("********  Distribute Error Calcs************")
        #for row in self.error_signal_calcs:
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
        converted_rows = [self.convert_numpy_scalars_because_python_is_weak(row) for row in self.error_signal_calcs]
        self.db.executemany(sql, self.error_signal_calcs)
        self.error_signal_calcs.clear()



    def validate_pass(self, target: float, prediction_raw: float):
        """
        Computes error, loss, and correction based on the configured loss function.
        """

        error = target - prediction_raw  # ✅ Simple error calculation
        loss = self.config.loss_function(prediction_raw, target)  # ✅ Compute loss dynamically
        if self.config.optimizer == "simplified_descent":
            correction = self.config.loss_function.grad(target, prediction_raw)  # ✅ Compute correction (NO inversion!)
            #print(f"🔎 DEBUG: Target={target}, Prediction={prediction_raw}, Error={error}, Loss={loss}, Correction={correction}")
        elif self.config.optimizer == "Simpletron":
            print("Warning: Simpletron has been deprecated")
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        return error, loss,  correction  # ✅ Correction replaces "loss gradient"

    def validate_pass_WithoutStrategy(self, target: float, prediction_raw:float):
            print("In new Validate")
            error = target - prediction_raw
            loss = error ** 2  # Example loss calculation (MSE for a single sample)
            loss_gradient = error * 2 #For MSE it is linear.
            #prediction =  1 if prediction_raw > .5 else 0      # Apply step function
            return error, loss,  loss_gradient

    ################################################################################################
    ################################ SECTION 3 - Initialization ####################################
    ################################################################################################

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
        for neuron in self.neurons:
            neuron.learning_rate = new_learning_rate


def smart_format(num):
    try:
        num = float(num)  # Ensure input is a number
    except (ValueError, TypeError):
        return str(num)  # If conversion fails, return as is

    if num == 0:
        return "0"
    #elif abs(num) < 1e-6:  # Use scientific notation for very small numbers
    #    return f"{num:.2e}"
    elif abs(num) < 0.001:  # Use 6 decimal places for small numbers
        #formatted = f"{num:,.6f}"
        return f"{num:.1e}"
    elif abs(num) < 1:  # Use 3 decimal places for numbers less than 1
        formatted = f"{num:,.3f}"
    elif abs(num) > 1000:  # Use no decimal places for large numbers
        formatted = f"{num:,.0f}"
    else:  # Default to 2 decimal places
        formatted = f"{num:,.2f}"



def store_num(number):
    formatted = f"{number:.6g}".replace(",", "")
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

def store_num(number):
    return number
