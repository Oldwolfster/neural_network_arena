import math
from typing import Tuple

from src.Legos.ActivationFunctions import *
from src.engine.BaseGladiator import Gladiator
from src.Legos.WeightInitializers import *
from src.Legos.LossFunctions import *
from src.engine.Config import Config
from src.engine.Neuron import Neuron
from src.engine.convergence.ConvergenceDetector import ROI_Mode

class NeuroForge_GoingBackToCali(Gladiator):
    """ ⚡implex: A ⚡imple Yet Powerful Neural Network ⚡
        ✅ Auto-tuned learning rate
        ✅ Supports multiple activation functions
        ✅ Flexible architecture with preconfigured alternatives
        🛡️ If you are having problems, comment everything out and try the 'smart defaults'
        """

    def configure_model(self, config: Config):
        """ 👉  Anything prior to initializing neurons goes here
            💪  For example setting config options.        """

        #Advice on California data
        # hidden_layers = [16, 8]
        # activation = "tanh"        output_activation = "linear"        loss_function = "mae"
        config.architecture         = [6,8]                       # Neurons in hidden layer output added automatically
        self.learning_rate          = .001 #If not included defaults to value in ArenaSettings
        config.initializer          = Initializer_Xavier
        config.output_activation    = Activation_NoDamnFunction
        config.loss_function        = Loss_MAE
        config.roi_mode             = ROI_Mode.MOST_ACCURATE    #SWEET_SPOT(Default), ECONOMIC or MOST_ACCURATE
        config.training_data        . set_normalization_z_score()

    def customize_neurons(self, config: Config):
        """ 🚀 Anything after initializing neurons
            🐉 but before training goes here  i.e manually setting a weight  """
        #Neuron.output_neuron.set_activation(Activation_NoDamnFunction)  #How to change a neurons activation initialization occured
        for layer in Neuron.layers[:-1]:  # Hidden layers
            for neuron in layer:
                neuron.bias= 525252


    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  RECOMMENDED FUNCTIONS TO CUSTOMIZE  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Remove not_running__ prefix to activate  🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Not running be default  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹    
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹


    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  Idiot proof features  🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹  THE KEY 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
    # 🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹

"""
1) Self setting LR
2) No expoding gradient
3) Does not allow incompatible output activtation function with loss functions
4) In fact, by default sets correct activation function for the loss function.


🥂   toasting
🐉   dragon
💪
🚀💯
🐍💥❤️
😈   devil
😂   laugh
⚙️   cog
🔍
🧠   brain
🥩   steak
"""