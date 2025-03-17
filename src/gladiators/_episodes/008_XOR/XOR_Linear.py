from src.engine.ActivationFunction import *
from src.engine.BaseGladiator import Gladiator
from src.engine.WeightInitializer import *
from src.Legos.LossFunctions import *
class MLP_Hayabusa(Gladiator):
    def __init__(self, *args):
        super().__init__(*args)
        self.initialize_neurons([1], [Initializer_Xavier], activation_function_for_hidden= Linear)
        self.neurons[0].activation = ReLU
        self.config.loss_function = Loss_BCEWithLogits


        #will add cost function soon.
        #will add optimizer option soon
        #what other options should be here?
        #Regularization  will be added soon

        #self.initialize_neurons([4,4,5],   activation_function_for_hidden= Tanh) #Test not specifying initializer
        #Neuron.layers(1).set_activation(ReLU)
        #self.neurons(1,1).set_activation(Tanh)
        #self.learning_rate = .1
        #self.initialize_neurons([2,3,5], [Initializer_He], ReLU)
        #self.initialize_neurons([10,10], [Initializer_He], ReLU)
        #self.initialize_neurons([2,3,5], [Initializer_Uniform], Tanh)
        #self.initialize_neurons([2,3,5,8,8,8], [Initializer_Uniform], Tanh)
        #self.initialize_neurons([2,4,2,5,2], [Initializer_Xavier], Tanh) #doesn't converge on seed 12345
