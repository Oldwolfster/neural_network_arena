from src.Legos.ActivationFunctions import *
from src.engine.BaseGladiator import Gladiator
from src.Legos.WeightInitializers import *
from src.Legos.LossFunctions import *

"""
Things to test.
1) SUCCESS: Overriding default methods.  
2) Single Neuron Binary Decision
3) Single Neuron Regrssion
4) MLP Regression
5) Adding Loss function functinoality
6) Seed 250490, epoch 2, iter 2,22 neuron 1-0, loss gradient looks wrong says - but should be positvie
"""
class MLP_Hayabusa(Gladiator):
    def __init__(self, config):
        config.loss_function = Loss_MSE
        super().__init__(config)
        self.initialize_neurons([], [Initializer_Xavier], activation_function_for_hidden= Tanh)
        #self.neurons[0].set_activation(Linear)


#print("🚀 NOT Using Default Forward pass - to customize override forward_pass")
