import math
from typing import Tuple

from src.Legos.ActivationFunctions import *
from src.Legos.Optimizers import Optimizer_Adam
from src.engine.BaseGladiator import Gladiator
from src.Legos.WeightInitializers import *
from src.Legos.LossFunctions import *
from src.engine.Neuron import Neuron
from src.engine.convergence.ConvergenceDetector import ROI_Mode

class MLP_Hayabusa(Gladiator):
    """ ⚡implex: A ⚡imple Yet Powerful Neural Network ⚡⚡⚡
        ✅ Auto-tuned learning rate
        ✅ Supports multiple activation functions
        ✅ Flexible architecture with preconfigured alternatives

        """
    def config_options(self, config) -> None:
        """ 🚀👉  Anything prior to calling superclass constructor) goes here
            💪 🐉 For example setting config options.        """
        config.architecture         = [4,4,4]                       # Neurons in hidden layer output added automatically
        self.learning_rate          = .01
        config.initializer          = Initializer_He
        config.output_activation    = Activation_Sigmoid
        #config.optimizer            = Optimizer_Adam
        config.hidden_activation     = Activation_LeakyReLU
        self.LR_Decay_rate = .5
        self.LR_Grow_rate = 1.05
        #config.training_data.set_normalization_min_max()
        #config.loss_function = Loss_MSE
        #config.roi_mode = ROI_Mode.MOST_ACCURATE       #SWEET_SPOT(Default), ECONOMIC or MOST_ACCURATE

    def initialize(self,config):             # 🚀 All additional initialization here

        self.initialize_neurons(
            architecture = [2],
            initializers = [Initializer_Xavier],
            hidden_activation = Activation_LeakyReLU,
            output_activation=Activation_NoDamnFunction
        )
        #Neuron.output_neuron.set_activation(Activation_NoDamnFunction)  #How to change a neurons activation initialization occured
        self.learning_rate = 4 #TODO silently f ails if called  before self.initalize_neurons
        #self.bd_threshold=0
        #self.bd_class_alpha=3

    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  RECOMMENDED FUNCTIONS TO CUSTOMIZE  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Remove not_running__ prefix to activate  🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Not running be default  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹

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