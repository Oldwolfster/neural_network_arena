from src.gladiators.BaseGladiator import Gladiator


class Hayabusa2(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.bias = .53
        self.weights = [0.1]
        self.prior_bias = 0 #other than initialized it will be 1 or -1
        self.bias_LR = self.learning_rate * .5


    def training_iteration(self, training_data) -> float:
        input = training_data[0]
        target = training_data[1]
        prediction  = input * self.weights[0] + self.bias           # Step 1) Guess
        error       = target - prediction                       # Step 2) Check guess,
        adjustment  = error * self.learning_rate                #         if wrong, how much
        self.weights[0]     += error * self.learning_rate                  # Step 3) Adjust(Apply)
        self.bias           += error * self.bias_LR
        self.check_bias_direction_change()
        return prediction


    def check_bias_direction_change(self):
        if self.metrics_mgr.iteration_num ==  1:
            print(f'first one - epoch {self.metrics_mgr.epoch_curr_number}\t bias={self.bias}')



