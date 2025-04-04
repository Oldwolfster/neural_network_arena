from src.Legos.ActivationFunctions import Tanh, Sigmoid
from src.engine.BaseGladiator import Gladiator

from src.engine.Neuron import Neuron
from src.engine.Utils import smart_format, store_num
from src.Legos.WeightInitializers import *


class XOR_TheLongWay(Gladiator):
    """
    This is my foray into MLP (Multi-Layer Perceptron) by solving the XOR problem,
    which cannot be linearly separated; hence, an SLP (Single-Layer Perceptron) just won't do.

    FORWARD PASS:
    Often, it is depicted as 5 neurons ([] below depicts a neuron):
        [Input1] | Both inputs connect | ==> \\    [Hidden1 (W11, W12, BiasH1)]   ===> [Output (WH1, WH2, B)]
        [Input2] | to both weights     | ==> //    [Hidden2 (W21, W22, BiasH2)]   ==/

    However, inputs are just part of the sample data, not really neurons themselves.
    So, we will declare 3 neurons: H1, H2, and O (hidden and output layers).

    Each hidden neuron (H1, H2) will be calculated as:
        tanh(W1 * Input1 + W2 * Input2 + BiasH1)

    The output neuron (O) will also apply the tanh activation function,
    but the final output will use a step function: if > 0, predict True; else, predict False.
    """

    def __init__(self, *args):
        super().__init__(*args)
        #self.set_random_seed(280312)   #Reuse an existing seed (to compare to past run)
        architecture = [2,4]  # Default: Single hidden layer with 2 neurons, 1 output

        self.initialize_neurons(architecture, [Initializer_Xavier])
        #self.learning_rate = 1 #must come after initialize_neurons
        """
        #Below is example of weights that make the model work right away
        #The XOR function outputs 1 when the inputs are different and 0 when they are the same. The truth table
        # Hidden Layer 1  #Neurons find roles.  for XOR one should be A or b
        self.neurons[0].weights[0] = 20.0  # Strong positive weight for first input
        self.neurons[0].weights[1] = 20.0  # Strong positive weight for second input
        self.neurons[0].bias = -10.0       # Negative bias to act as threshold

        # Hidden Layer 2  #'not (a and b)'
        self.neurons[1].weights[0] = -20.0  # Strong negative weight for first input
        self.neurons[1].weights[1] = -20.0  # Strong negative weight for second input
        self.neurons[1].bias  =30.0         # Large positive bias

        # Output Layer
        self.neurons[2].weights[0] = 20.0   # Strong positive weight from H1
        self.neurons[2].weights[1] = 20.0   # Strong positive weight from H2
        self.neurons[2].bias = -10.0        # Negative bias
        
        
        # Initialize weights with small, non-symmetric values - BUT WILL NOT CONVERGE.  Will saturate
        self.neurons[0].weights[0] = 0.1
        self.neurons[0].weights[1] = 0.2
        self.neurons[0].bias = 1

        self.neurons[1].weights[0] = 0.3
        self.neurons[1].weights[1] = 0.4
        self.neurons[1].bias = 2

        self.neurons[2].weights[0] = 0.5
        self.neurons[2].weights[1] = 0.6
        self.neurons[2].bias = 3
        

                # Hidden Layer 1
        self.neurons[0].weights[0] = 0.5  # Small positive weight for x1
        self.neurons[0].weights[1] = 0.5  # Small positive weight for x2
        self.neurons[0].bias = -0.2       # Slightly negative bias to act as threshold

        self.neurons[1].weights[0] = -0.5 # Small negative weight for x1
        self.neurons[1].weights[1] = -0.5 # Small negative weight for x2
        self.neurons[1].bias = 0.2        # Slightly positive bias

        # Output Layer
        self.neurons[2].weights[0] = 0.5  # Small positive weight from H1
        self.neurons[2].weights[1] = 0.5  # Small positive weight from H2
        self.neurons[2].bias = -0.2       # Slightly negative bias
        """



        self.output_tanh            = 0         # Variables we need from forward prop to do back prop
        self.hidden_1_output        = 0         # Variables we need from forward prop to do back prop
        self.hidden_2_output        = 0         # Variables we need from forward prop to do back prop
        self.neurons[0].set_activation(Tanh)        # Set activation to Tanh on neuron 1,0
        self.neurons[1].set_activation(Tanh)        # Set activation to Tanh on neuron 1,1
        self.neurons[2].set_activation(Sigmoid)     # Set activation to Sig on neuron 2,0

    def back_pass(self, training_sample, loss_gradient: float):
        input_0 = training_sample[0]  # First input
        input_1 = training_sample[1]  # Second input
        target  = training_sample[-1]  # Target value
        output_neuron = Neuron.layers[-1][0]

        # Step 1: Compute error signal for output neuron
        self.back_pass__error_signal_for_output(loss_gradient)

        # Step 2: Compute error signals for hidden neurons
        for hidden_neuron in Neuron.layers[0]:  # Iterate over first hidden layer
            self.back_pass__error_signal_for_hidden( hidden_neuron)

        # Step 3: Adjust weights for the output neuron
        prev_layer_activations = [n.activation_value for n in Neuron.layers[-2]]  # Last hidden layer activations
        self.back_pass_distribute_error(output_neuron, prev_layer_activations)


        # Step 4: Adjust weights for the hidden neurons (⬅️ Last step we need)
        for layer_index in range(len(Neuron.layers)       - 2, 0, -1):  # Iterate backwards (excluding input layer)
            prev_layer_activations = [n.activation_value for n in Neuron.layers[layer_index - 1]]
            for neuron in Neuron.layers[layer_index]:
                self.back_pass_distribute_error(neuron, prev_layer_activations)

        for neuron in Neuron.layers[0]:  # First hidden layer
            self.back_pass_distribute_error(neuron, training_sample[:-1])  # Use raw inputs

    def back_pass__error_signal_for_hidden(self, to_neuron: Neuron):
        """
        Calculate the error signal for a hidden neuron by summing the contributions from all neurons in the next layer.
        """
        activation_gradient = to_neuron.activation_gradient
        total_backprop_error = 0  # Sum of (next neuron error * connecting weight)
        to_neuron.blame_calculations=""

        #print(f"Calculating error signal epoch/iter:{self.epoch}/{self.iteration} for neuron {to_neuron.layer_id},{to_neuron.position}")
        # 🔄 Loop through each neuron in the next layer
        for next_neuron in Neuron.layers[to_neuron.layer_id + 1]:  # Next layer neurons
            #print (f"getting weight and error from {to_neuron.layer_id},{to_neuron.position}")
            weight_to_next = next_neuron.weights_before[to_neuron.position]  # Connection weight
            error_from_next = next_neuron.error_signal  # Next neuron’s error signal
            total_backprop_error += weight_to_next * error_from_next  # Accumulate contributions
            to_neuron.blame_calculations= to_neuron.blame_calculations + f"{smart_format( weight_to_next)}!{smart_format( error_from_next)}@"

        #print (f"yoooo{to_neuron.blame_calculations}")
        # 🔥 Compute final error signal for this hidden neuron
        to_neuron.error_signal = activation_gradient * total_backprop_error


    def back_pass__error_signal_for_hiddenOld(self, from_neuron : Neuron, to_neuron: Neuron):
        # Formula -> Activation gradient * Sum(Next Layer weight * Next neuron  Error signal)
        # NOTE from_neuron is to the right because it's going backwards
        #In the case of single output there is only one value to sum, the output neuron

        #print(f"\n🔄 Propagating Blame from Layer {from_neuron.layer_id} to Layer {to_neuron.layer_id}, Neuron ID: {to_neuron.nid}")

        activation_gradient = to_neuron.activation_gradient
        weight_index = to_neuron.position
        from_neuron_weight = from_neuron.weights_before[weight_index]
        from_neuron_error_signal = from_neuron.error_signal
        to_neuron.error_signal = activation_gradient * from_neuron_weight * from_neuron_error_signal
        #print(f"calculating error_signal for neuron: {to_neuron.layer_id}, {to_neuron.position}\n"
        #      f"activation_gradient\t{activation_gradient}\n"
        #      f"from neuron weight\t{from_neuron_weight}\n"
        #      f"from neuron err sig\t{from_neuron_error_signal}\n"
        #      f"equals {to_neuron.error_signal}")



    def back_pass__determine_blame_for_output_neuron(self, loss_gradient: float):
        output_neuron = Neuron.layers[-1][0]
        activation_gradient = output_neuron.activation_gradient
        error_signal = loss_gradient * activation_gradient
        output_neuron.error_signal = error_signal

    def back_pass_distribute_error(self, neuron: Neuron, prev_layer_values):
        """
        Updates weights for a neuron based on error signal.

        - First hidden layer uses inputs from training data.
        - All other neurons use activations from the previous layer.
        """

        learning_rate = neuron.learning_rate
        error_signal = neuron.error_signal

        weight_formulas = []

        # FORMULA: weight_new = weight_old + (learning_rate * error_signal * previous_layer_value)
        #print(f"Weights{neuron.weights}\tprev_layer_values{prev_layer_values}")

        #print(f"Weights_before (before building weight formula) { Neuron.layers[1][0].weights_before}")
        for i, (w, prev_value) in enumerate(zip(neuron.weights, prev_layer_values)):
            neuron.weights[i] += learning_rate * error_signal * prev_value
            neuron_id = f"{neuron.layer_id},{neuron.position}"
            #calculation = f"w{i} Neuron ID{neuron_id} = {smart_format(w)} + {smart_format(learning_rate)} * {smart_format(error_signal)} * {smart_format(prev_value)}"
            calculation = f"w{i} Neuron ID{neuron_id} = {store_num(w)} + {store_num(learning_rate)} * {store_num(error_signal)} * {store_num(prev_value)}"
            weight_formulas.append(calculation)

        # Bias update
        neuron.bias += learning_rate * error_signal
        weight_formulas.append(f"B = {store_num(neuron.bias_before)} + {store_num(learning_rate)} * {store_num(error_signal)}")
        neuron.weight_adjustments = '\n'.join(weight_formulas)



    def forward_pass(self, training_sample):
        """
        Manually computes forward pass for each neuron in the XOR MLP.

        :param input_0: First input feature
        :param input_1: Second input feature
        :return: prediction (final output of the network)
        """
        input_0 = training_sample[0]  # First input
        input_1 = training_sample[1]  # Second input
        target  = training_sample[-1]  # Target value

        # 🔹 Inputs are explicitly provided
        input_values = [input_0, input_1]

        # 🚀 Compute raw sums + activations for first hidden layer (2 neurons)
        self.neurons[0].raw_sum = (
            (input_values[0] * self.neurons[0].weights[0]) +
            (input_values[1] * self.neurons[0].weights[1]) +
            self.neurons[0].bias
        )
        self.neurons[0].activate()

        self.neurons[1].raw_sum = (
            (input_values[0] * self.neurons[1].weights[0]) +
            (input_values[1] * self.neurons[1].weights[1]) +
            self.neurons[1].bias
        )
        self.neurons[1].activate()

        # 🚀 Compute raw sum + activation for Output neuron (Sigmoid)
        self.neurons[2].raw_sum = (
            (self.neurons[0].activation_value * self.neurons[2].weights[0]) +
            (self.neurons[1].activation_value * self.neurons[2].weights[1]) +
            self.neurons[2].bias
        )
        self.neurons[2].activate()

        # DEBUGGING:
        #print(f"Hidden Neuron 0: raw_sum={self.neurons[0].raw_sum}, activation_value={self.neurons[0].activation_value}")
        #print(f"Hidden Neuron 1: raw_sum={self.neurons[1].raw_sum}, activation_value={self.neurons[1].activation_value}")
        output_raw =  self.neurons[0].activation_value  * self.neurons[2].weights[0]  + self.neurons[1].activation_value * self.neurons[2].weights[1] + self.neurons[2].bias
        #print(f"output_raw===({self.neurons[0].activation_value} * {self.neurons[2].weights[0]} + {self.neurons[1].activation_value} * {self.neurons[2].weights[1]} + {self.neurons[2].bias} = {output_raw} <==DOES IT???)")
        #print(f"Output Neuron: raw_sum={self.neurons[2].raw_sum}, activation_value={self.neurons[2].activation_value}")

        return self.neurons[2].activation_value  # 🚀 Final prediction


