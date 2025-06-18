import math
from typing import Tuple

from src.Legos.ActivationFunctions import *
from src.NNA.engine.BaseGladiator import Gladiator
from src.Legos.WeightInitializers import *
from src.Legos.LossFunctions import *
from src.Legos.Optimizers import *
from src.NNA.engine.Config import Config
from src.NNA.engine.Neuron import Neuron
from src.NNA.engine.convergence.ConvergenceDetector import ROI_Mode

class Adam_XOR_debug(Gladiator):
    """ ⚡implex: A ⚡imple Yet Powerful Neural Network ⚡
        ✅ Auto-tuned learning rate
        ✅ Supports multiple activation functions
        ✅ Flexible architecture with preconfigured alternatives
        🛡️ If you are having problems, comment everything out and try the 'smart defaults'
    """

    def configure_model(self, config: Config):
        """ 👉  Anything prior to initializing neurons goes here
            💪  For example setting config options.

                Below weights converged in 75 epochs in pytorch FORMAT:
                h0 .497585, -.5147896 with bias of -0.2578
                [(array([[ 0.497585  , -0.17507666],  [-0.5147896 ,  0.04176939]], dtype=float32), array([-0.25778118, -0.6163607 ], dtype=float32)), (array([[-0.5352204 , -0.04399097]], dtype=float32), array([-0.24767366], dtype=float32))]


                    """

        config.architecture         = [2]  # 2 hidden neurons
        config.optimizer            = Optimizer_Adam
        config.loss_function        = Loss_MSE
        config.learning_rate        = 0.01  # Matches PyTorch LR
        #hardcoded - config.optimizer_beta1      = 0.5
        #hardcoded - config.optimizer_beta2      = 0.9
        config.initializer          = Initializer_Xavier
        config.hidden_activation    = Activation_Tanh
        config.output_activation    = Activation_NoDamnFunction  # Linear


    def customize_neurons(self, config: Config):
        """ 🚀 Anything after initializing neurons
            🐉 but before training goes here  i.e manually setting a weight  """

        # PyTorch weights and biases manually transferred from the trained model:
        # Input to hidden layer weights
        Neuron.layers[0][0].weights  = [0.497585, -0.17507666]  # Hidden neuron 0
        Neuron.layers[0][1].weights  = [-0.5147896, 0.04176939]  # Hidden neuron 1
        Neuron.layers[0][0].bias     = -0.25778118
        Neuron.layers[0][1].bias     = -0.6163607

        # Hidden to output layer weights
        Neuron.layers[1][0].bias     = -0.24767366
        Neuron.layers[1][0].weights  = [-0.5352204, -0.04399097]



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