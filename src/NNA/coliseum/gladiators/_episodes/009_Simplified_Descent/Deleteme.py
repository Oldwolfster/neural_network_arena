from src.Legos.ActivationFunctions import *
from src.NNA.engine.BaseGladiator import Gladiator
from src.Legos.WeightInitializers import *
from src.Legos.LossFunctions import *
from src.NNA.engine.Neuron import Neuron

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

        config.loss_function = Loss_MSE
        super().__init__(config)
        self.initialize_neurons([], [Initializer_Xavier]
                                , hidden_activation= Activation_Tanh)


        self.learning_rate = 4 #TODO silently f ails if called  before self.initalize_neurons
        #self.bd_threshold=0
        #self.bd_class_alpha=3
        Neuron.output_neuron.set_activation(Activation_NoDamnFunction)


    def back_pass__distribute_error(self, neuron: Neuron, prev_layer_values):
        """
        Updates weights for a neuron based on blame (error signal).
        args: neuron: The neuron that will have its weights updated to.

        - First hidden layer uses inputs from training data.
        - All other neurons use activations from the previous layer.
        """
        error_signal = neuron.error_signal
        if self.convergence_phase == "fix":
            if self.LR_Grow_rate>1:
                self.LR_Grow_rate = .99
            self.LR_Grow_rate *= .999999
            print (f"self.LR_Grow_rate= {self.LR_Grow_rate}")

        for i, (w, prev_value) in enumerate(zip(neuron.weights, prev_layer_values)):
            weight_before = neuron.weights[i]
            adjustment  = prev_value * error_signal *  neuron.learning_rates[i+1] #1 accounts for bias in 0  #So stupid to go down hill they look uphill and go opposite
            if abs(adjustment) > self.too_high_adjst: #Explosion detection
                adjustment = 0
                neuron.learning_rates[i+1] *= self.LR_Decay_rate     #reduce neurons LR
            # **💡 Growth Factor: Gradually Increase LR if too slow**

            #elif not is_exploding(weight) and not is_oscillating(weight):
            else:
                neuron.learning_rates[i] *= self.LR_Grow_rate  # Boost LR slightly if it looks stable

            neuron.weights[i] -= adjustment
            #print(f"trying to find path down{self.epoch+1}, {self.iteration+1}\tprev_value{prev_value}\terror_signal{error_signal}\tlearning_rate{learning_rate}\tprev_value{adjustment}\t")

            # 🔹 Store structured calculation for weights
            self.weight_calculations.append([
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
        self.weight_calculations.append([
        # epoch, iteration, model_id, neuron_id, weight_index, arg_1, op_1, arg_2, op_2, arg_3, op_3, result
            self.epoch+1 , self.iteration+1, self.gladiator, neuron.nid, 0,
                "1", "*", error_signal, "*", neuron.learning_rates[0],   "=", adjustment_bias
            ])
