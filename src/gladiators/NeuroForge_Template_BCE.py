from src.Legos.ActivationFunctions import *
from src.engine.BaseGladiator import Gladiator
from src.Legos.WeightInitializers import *
from src.Legos.LossFunctions import *
from src.engine.Neuron import Neuron

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
        config.loss_function = Loss_MSE
        super().__init__(config)
        self.initialize_neurons([2], [Initializer_Xavier], activation_function_for_hidden= Activation_Tanh)
        #self.bd_threshold=0
        #self.bd_class_alpha=3
        Neuron._output_neuron.set_activation(Activation_NoDamnFunction)


#print("🚀 NOT Using Default Forward pass - to customize override forward_pass")
