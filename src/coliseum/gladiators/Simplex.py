import math
from typing import Tuple

from src.Legos.ActivationFunctions import *
from src.engine.BaseGladiator import Gladiator
from src.Legos.WeightInitializers import *
from src.Legos.LossFunctions import *
from src.engine.Config import Config
from src.engine.Neuron import Neuron
from src.engine.convergence.ConvergenceDetector import ROI_Mode

class NeuroForge_Template(Gladiator):
    """ ⚡implex: A ⚡imple Yet Powerful Neural Network ⚡
        ✅ Auto-tuned learning rate
        ✅ Supports multiple activation functions
        ✅ Flexible architecture with preconfigured alternatives
        👉 If you are having problems, comment everything out and try the 'smart defaults'
        """

    def configure_model(self, config: Config):
        """ 👉  Anything prior to initializing neurons goes here
            💪  For example setting config options.        """

        config.architecture             = [2]                       # Neurons in hidden layer output added automatically
        self.learning_rate              = 4
        self.config.loss_function       = Loss_MSE
        self.config.initializer         = Initializer_Xavier
        self.config.output_activation   = Activation_NoDamnFunction
        self.config.hidden_activation   = Activation_Tanh


        self.LR_Decay_rate = .5
        self.LR_Grow_rate = 1.05
        #config.roi_mode             = ROI_Mode.MOST_ACCURATE    #SWEET_SPOT(Default), ECONOMIC or MOST_ACCURATE
        #config.training_data        . set_normalization_min_max()

    def customize_neurons(self, config: Config):
        """ 🚀 Anything after initializing neurons
            🐉 but before training goes here  i.e manually setting a weight  """
        #Neuron.output_neuron.set_activation(Activation_NoDamnFunction)  #How to change a neurons activation initialization occured


    def back_pass__update_neurons_weights(self, neuron: Neuron, prev_layer_values: list[float]) -> None:
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
