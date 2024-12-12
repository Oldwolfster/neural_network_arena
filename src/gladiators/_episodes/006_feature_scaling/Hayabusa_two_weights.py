from src.gladiators.BaseGladiator import Gladiator


class SuzukiHayabusaTwoWeights(Gladiator):
    """
    A simple single input regression model
    This version will utilize both weights
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.initialize_neurons(2)
        #self.training_data.set_normalization_min_max()

    def training_iteration(self, training_data) -> float:
        inp_0            = training_data[0]                     # First Element of each tuple
        inp_1            = training_data[1]                     # First Element of each tuple
        target           = training_data[-1]                    # Last element is target

        prediction       = inp_0 * self.weights[0] + self.bias + inp_1 * self.weights[1] # Step 1) Guess
        error            = target - prediction                  # Step 2) Check guess
        # print (error)
        self.weights[0] += error * self.learning_rate * inp_0   # Step 3) Adjust Weight - formula for gradient descent
        self.weights[1] += error * self.learning_rate * inp_1   # Step 3) Adjust Weight - formula for gradient descent
        self.bias       += error * self.learning_rate           # Step 3) Adjust Bias
        return prediction

