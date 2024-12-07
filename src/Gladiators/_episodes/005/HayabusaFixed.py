import numpy as np

from src.gladiators.BaseGladiator import Gladiator


class SuzukiHayabusaNoExplode(Gladiator):
    """
    A simple single input regression model
    This version will utilize both weights
    Note: self.metrics_mgr is created in base class Gladiator and has info available
    """

    def __init__(self, *args):
        super().__init__(*args)

    def training_iteration(self, training_data) -> float:
        inp_0            = training_data[0]                     # First Element of each tuple
        inp_1            = training_data[1]                     # First Element of each tuple
        target           = training_data[-1]                    # Last element is target
        self.metrics_mgr
        prediction       = inp_0 * self.weights[0] + self.bias + inp_1 * self.weights[1] # Step 1) Guess
        error            = target - prediction                  # Step 2) Check guess
        #print (error)
        self.weights[0] += error * self.learning_rate * inp_0   # Step 3) Adjust Weight - formula for gradient descent
        self.weights[1] += error * self.learning_rate * inp_1   # Step 3) Adjust Weight - formula for gradient descent

        gradient = error * inp_1
        threshold = 1
        clipped_gradient = np.clip(gradient, -threshold, threshold)

        gradient_1 = error * inp_0
        gradient_2 = error * inp_1

        #print(f"Epoch {self.metrics_mgr.epoch_curr_number}: Gradient 1: {gradient_1:.6f}, Gradient 2: {gradient_2:.6f}")

        #self.weights[1] += self.learning_rate * clipped_gradient
        self.bias       += error * self.learning_rate           # Step 3) Adjust Bias
        return prediction