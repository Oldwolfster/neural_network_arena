import math

from src.Legos.ActivationFunctions import *
from src.NNA.engine.BaseGladiator import Gladiator
from src.Legos.WeightInitializers import *
from src.Legos.LossFunctions import *
from src.NNA.engine.Neuron import Neuron
from src.NNA.engine.convergence.ConvergenceDetector import ROI_Mode

"""
Things to test.
1) SUCCESS: Overriding default methods.  
2) Single Neuron Binary Decision
3) Single Neuron Regrssion
4) MLP Regression
5) Adding Loss function swappability
6) Seed 250490, epoch 2, iter 2,22 neuron 1-0, loss gradient looks wrong says - but should be positvie
"""
class MLP_Hayabusa(Gladiator):
    """
        ⚡ MLP_Hayabusa: A Simple Yet Powerful Neural Network ⚡

        ✅ Auto-tuned learning rate
        ✅ Supports multiple activation functions
        ✅ Flexible architecture with preconfigured alternatives

        🔹 Change anything below to customize your model!
        """


    def __init__(self, config):
        self.LR_Decay_rate = .5
        self.LR_Grow_rate = 1.05
        config.training_data.set_normalization_min_max()
        config.loss_function = Loss_MSE
        config.roi_mode = ROI_Mode.MOST_ACCURATE       #SWEET_SPOT(Default), ECONOMIC or MOST_ACCURATE
        super().__init__(config)
        self.initialize_neurons([2], [Initializer_Xavier]
                                , hidden_activation= Activation_LeakyReLU)

        self.learning_rate = .00000001 #TODO silently f ails if called  before self.initalize_neurons
        #self.bd_threshold=0
        #self.bd_class_alpha=3
        Neuron.output_neuron.set_activation(Activation_LeakyReLU)


    def back_pass__distribute_error3(self, neuron: Neuron, prev_layer_values):
        """
        Updates weights and bias for a neuron using blame (error signal).
        - First hidden layer uses inputs from training data.
        - All other neurons use activations from the previous layer.
        """
        error_signal = neuron.error_signal

        if self.convergence_phase == "fix":
            if self.LR_Grow_rate > 1:
                self.LR_Grow_rate = 0.99
            self.LR_Grow_rate *= 0.999999
            print(f"self.LR_Grow_rate= {self.LR_Grow_rate}")

        # Construct [bias_input (1.0), *prev_layer_values] to align with learning_rates[0:] and weights[0:]
        input_vector = [1.0] + list(prev_layer_values)

        for i, prev_value in enumerate(input_vector):
            learning_rate = neuron.learning_rates[i]
            adjustment = prev_value * error_signal * learning_rate

            # Explosion detection
            if abs(adjustment) > self.too_high_adjst:
                adjustment = 0
                neuron.learning_rates[i] *= self.LR_Decay_rate
            else:
                neuron.learning_rates[i] *= self.LR_Grow_rate

            # Cap adjustment if needed
            if abs(adjustment) > abs(self.max_adj):
                adjustment = math.copysign(1, adjustment) * self.max_adj

            # Update
            if i == 0:
                neuron.bias -= adjustment
            else:
                neuron.weights[i - 1] -= adjustment

            # Track structured calc
            self.weight_calculations.append([
                self.epoch + 1, self.iteration + 1, self.gladiator, neuron.nid, i,
                prev_value, "*", error_signal, "*", learning_rate, "=", adjustment
            ])
