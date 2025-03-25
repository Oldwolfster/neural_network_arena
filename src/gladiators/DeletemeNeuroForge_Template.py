from src.engine.BaseGladiator       import Gladiator
from src.Legos.ActivationFunctions  import *
from src.Legos.WeightInitializers   import *
from src.Legos.LossFunctions        import *
from src.engine.Neuron              import Neuron
from src.engine.convergence.ConvergenceDetector import ROI_Mode


class Simplex(Gladiator):     #Model selection goverened by file name
    """ ⚡ Simplex: A Simple Yet Powerful Neural Network ⚡
        ✅ Auto-tuned learning rate
        ✅ Supports multiple activation functions
        ✅ Flexible architecture with preconfigured alternatives
        """
    def config_options(self, config) -> None:
        """ 🚀  Anything prior to calling superclass constructor) goes here
                For example setting config options.
        """
        config.loss_function = Loss_MSE
        config.roi_mode = ROI_Mode.MOST_ACCURATE       #SWEET_SPOT(Default), ECONOMIC or MOST_ACCURATE
        config.training_data.set_normalization_min_max()

    def initialize(self,config):             # 🚀 All additional initialization here
        self.initialize_neurons(architecture = [2],
                                initializers = [Initializer_Xavier],
                                activation_function_for_hidden = Activation_LeakyReLU)

        Neuron.output_neuron.set_activation(Activation_NoDamnFunction)

        #self.bd_threshold      = 0         # If Binary Decision modify threshold (default = 0)
        #self.bd_class_alpha    = 0         # If Binary Decision set class alpha  (default = 0)
        #self.bd_class_beta     = 1         # If Binary Decision set class beta   (default = 1)

    def __init__(self,       config):   #   Constructor for Gladiator
        self.config_options (config)    #   Set config options below
        super().__init__    (config)    #   Constructor for superclass and initializer
        self.initialize     (config)    #   Perform initialization below

    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  ABOVE HERE IS ALL BOILERPLATE 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  CUSTOMIZE BELOW!!             🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹


#print("🚀 NOT Using Default Forward pass - to customize override forward_pass")
