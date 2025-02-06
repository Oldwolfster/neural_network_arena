import traceback

import numpy as np

from src.engine.ActivationFunction import *
from src.engine.BaseGladiator import Gladiator
import math

from src.engine.Neuron import Neuron
from src.engine.Utils import smart_format, print_call_stack, store_num
from src.engine.WeightInitializer import *


class MLP_Hayabusa(Gladiator):
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
        self.initialize_neurons([3,4], [Initializer_Xavier], Tanh)
        #self.initialize_neurons([2,4,2,5,2], [Initializer_Xavier], Tanh) #doesn't converge on seed 12345
    def forward_pass(self, training_sample):
            """
            Computes forward pass for each neuron in the XOR MLP.
            """
            input_values = training_sample[:-1]

            # 🚀 Compute raw sums + activations for each layer
            for layer_idx, layer in enumerate(Neuron.layers):  # Exclude output layer
                prev_activations = input_values if layer_idx == 0 else [n.activation_value for n in Neuron.layers[layer_idx - 1]]

                for neuron in layer:
                    neuron.raw_sum = sum(input_val * weight for input_val, weight in zip(prev_activations, neuron.weights))
                    neuron.raw_sum += neuron.bias
                    neuron.activate()

    def back_pass(self, training_sample, loss_gradient: float):
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
        prev_layer_activations = [n.activation_value for n in Neuron.layers[-2]]  # Last hidden layer activations
        self.back_pass_distribute_error(output_neuron, prev_layer_activations)

        # Step 4: Adjust weights for the hidden neurons (⬅️ Last step we need)
        for layer_index in range(len(Neuron.layers) - 2, -1, -1):  # Iterate backwards (including first hidden layer)
            prev_layer_activations = [n.activation_value for n in Neuron.layers[layer_index - 1]]  # Use activations for hidden layers
            if layer_index == 0:        #For layer zero overwrite prev_layer_activations with inputs as inputs aren't in the neuron layers.
                prev_layer_activations = training_sample[:-1]  # Use raw inputs for first hidden layer
            for neuron in Neuron.layers[layer_index]:
                self.back_pass_distribute_error(neuron, prev_layer_activations)

    def back_pass__error_signal_for_output(self, loss_gradient: float):
        """
        Calculate error_signal(gradient) for output neuron.
        Assumes one output neuron and that loss_gradient has already been calculated.
        """
        output_neuron               = Neuron.layers[-1][0]
        activation_gradient         = output_neuron.activation_gradient
        error_signal                = loss_gradient * activation_gradient
        output_neuron.error_signal  = error_signal

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
            #OLD WAY neuron.error_signal_calcs= neuron.error_signal_calcs + f"{smart_format( weight_to_next)}!{smart_format( error_from_next)}@"
            memory_efficent_way_to_store_calcs.append(f"{smart_format(weight_to_next)}!{smart_format(error_from_next)}@")
        neuron.error_signal_calcs = ''.join(memory_efficent_way_to_store_calcs)  # Join once instead of multiple string concatenations


        # 🔥 Compute final error signal for this hidden neuron
        neuron.error_signal = activation_gradient * total_backprop_error


    def back_pass_distribute_error(self, neuron: Neuron, prev_layer_values):
        """
        Updates weights for a neuron based on error signal.
        args: neuron: The neuron that will have its weights updated to.

        - First hidden layer uses inputs from training data.
        - All other neurons use activations from the previous layer.
        """
        learning_rate = neuron.learning_rate
        error_signal = neuron.error_signal
        weight_formulas = []

        for i, (w, prev_value) in enumerate(zip(neuron.weights, prev_layer_values)):
            neuron.weights[i] += learning_rate * error_signal * prev_value
            neuron_id = f"{neuron.layer_id},{neuron.position}"
            calculation = f"w{i} Neuron ID{neuron_id} = {store_num(w)} + {store_num(learning_rate)} * {store_num(error_signal)} * {store_num(prev_value)}"
            weight_formulas.append(calculation)

        # Bias update
        neuron.bias += learning_rate * error_signal
        weight_formulas.append(f"B = {store_num(neuron.bias_before)} + {store_num(learning_rate)} * {store_num(error_signal)}")
        neuron.weight_adjustments = '\n'.join(weight_formulas)
