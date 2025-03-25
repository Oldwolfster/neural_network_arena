import math
from typing import Tuple

from src.Legos.ActivationFunctions import *
from src.engine.BaseGladiator import Gladiator
from src.Legos.WeightInitializers import *
from src.Legos.LossFunctions import *
from src.engine.Neuron import Neuron
from src.engine.convergence.ConvergenceDetector import ROI_Mode

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
        config.training_data.set_normalization_min_max()
        #config.loss_function = Loss_MSE
        config.roi_mode = ROI_Mode.MOST_ACCURATE       #SWEET_SPOT(Default), ECONOMIC or MOST_ACCURATE

    def initialize(self,config):             # 🚀 All additional initialization here

        self.initialize_neurons(
            architecture = [3,3,3],
            initializers = [Initializer_Xavier],
            #hidden_activation = Activation_Tanh,
            # Great idea output_activation = Activation_LeakyReLU
        )
        #Neuron.output_neuron.set_activation(Activation_NoDamnFunction)  #How to change a neurons activation initialization occured
        self.learning_rate = .001 #TODO silently f ails if called  before self.initalize_neurons
        #self.bd_threshold=0
        #self.bd_class_alpha=3


    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  RECOMMENDED FUNCTIONS TO CUSTOMIZE  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Remove not_running__ prefix to activate  🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Not running be default  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
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