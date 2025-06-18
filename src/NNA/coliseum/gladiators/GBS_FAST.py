import math
from typing import Tuple

from src.Legos.ActivationFunctions import *
from src.NNA.engine.BaseGladiator import Gladiator
from src.Legos.WeightInitializers import *
from src.Legos.LossFunctions import *
from src.NNA.engine.Neuron import Neuron
from src.NNA.engine.convergence.ConvergenceDetector import ROI_Mode

class MLP_Hayabusa(Gladiator):
    """ ⚡implex: A ⚡imple Yet Powerful Neural Network ⚡
        ✅ Auto-tuned learning rate
        ✅ Supports multiple activation functions
        ✅ Flexible architecture with preconfigured alternatives

        """
    def config_options(self, config) -> None:
        """ 🚀👉  Anything prior to calling superclass constructor) goes here
            💪 🐉 For example setting config options.        """

        self.LR_Decay_rate = .5
        self.LR_Grow_rate = 1.05
        #config.training_data.set_normalization_min_max()
        #config.loss_function = Loss_MSE
        config.roi_mode = ROI_Mode.MOST_ACCURATE       #SWEET_SPOT(Default), ECONOMIC or MOST_ACCURATE

    def initialize(self,config):             # 🚀 All additional initialization here

        self.initialize_neurons(
            architecture = [2],
            initializers = [Initializer_Xavier],
            #hidden_activation = Activation_Tanh,
            # Great idea output_activation = Activation_LeakyReLU
        )
        #Neuron.output_neuron.set_activation(Activation_NoDamnFunction)  #How to change a neurons activation initialization occured
        self.learning_rate = .002 #TODO silently f ails if called  before self.initalize_neurons
        #self.bd_threshold=0
        #self.bd_class_alpha=3


    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  RECOMMENDED FUNCTIONS TO CUSTOMIZE  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Remove not_running__ prefix to activate  🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Not running be default  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    def not_running__forward_pass(self, training_sample: Tuple[float, float, float]) -> None:
        """
        🚀 Computes forward pass for each neuron in the XOR MLP.
        🔍 Activation of Output neuron will be considered 'Raw Prediction'
        Args:
            training_sample: tuple where first elements are inputs and last element is target (assume one target)
        """
        input_values = training_sample[:-1]

        # 🚀 Compute raw sums + activations for each layer
        for layer_idx, layer in enumerate(Neuron.layers):  # Exclude output layer
            prev_activations = input_values if layer_idx == 0 else [n.activation_value for n in Neuron.layers[layer_idx - 1]]

            for neuron in layer:
                neuron.raw_sum = sum(input_val * weight for input_val, weight in zip(prev_activations, neuron.weights))
                neuron.raw_sum += neuron.bias
                neuron.activate()

    def not_running__back_pass(self, training_sample : Tuple[float, float, float], loss_gradient: float):
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

    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹 NOTE:  All below are helpers for back_pass.(above)🔹🔹🔹 🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹 If you COMPLETELY replace backpass they are not needed.  🔹🔹🔹🔹 🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    def not_running__back_pass__determine_blame_for_output_neuron(self, loss_gradient: float):
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

    def not_running__back_pass__determine_blame_for_a_hidden_neuron(self, neuron: Neuron):
        """
        Calculate the error signal for a hidden neuron by summing the contributions from all neurons in the next layer.
        args: neuron:  The neuron we are calculating the error for.
        """
        activation_gradient         = neuron.activation_gradient
        total_backprop_error        = 0  # Sum of (next neuron error * connecting weight)

        # 🔄 Loop through each neuron in the next layer
        for next_neuron in Neuron.layers[neuron.layer_id + 1]:  # Next layer neurons
            weight_to_next          =  next_neuron.weights_before[neuron.position]  # Connection weight
            error_from_next         =  next_neuron.error_signal  # Next neuron’s error signal
            total_backprop_error    += weight_to_next * error_from_next  # Accumulate contributions
            neuron.error_signal     =  activation_gradient * total_backprop_error # 🔥 Compute final error signal for this hidden neuron

            # 🔹 Store calculation step as a structured tuple, now including weight index
            self.blame_calculations.append([
                self.epoch+1, self.iteration+1, self.gladiator, neuron.nid, next_neuron.position,
                weight_to_next, "*", error_from_next, "=", None, None, weight_to_next * error_from_next
            ])

    def not_running__back_pass__spread_the_blame(self, training_sample : Tuple[float, float, float]):
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

    def not_running__back_pass__update_neurons_weights(self, neuron: Neuron, prev_layer_values: list[float]) -> None:
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
            self.weight_calculations.append([
                self.epoch + 1, self.iteration + 1, self.gladiator, neuron.nid, i,
                prev_value, "*", blame, "*", learning_rate, "=", adjustment
            ])

    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  BELOW HERE IS ALL BOILERPLATE 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  RECOMMEND TO CUSTOMIZE ABOVE! 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    def __init__(self, config):
        self.config_options(config)
        super().__init__(config)
        self.initialize(config)

    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Idiot proof features  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  THE KEY 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹

"""
1) Self setting LR
2) No expoding gradient
3) Does not allow incompatible output activtation function with loss functions
4) In fact, by default sets correct activation function for the loss function.


🥂 toasting

"""