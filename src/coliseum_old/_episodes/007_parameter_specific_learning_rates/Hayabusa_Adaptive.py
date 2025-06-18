from src.NNA.engine.BaseGladiator import Gladiator


class SuzukiHayabusa(Gladiator):
    """
    A simple single input regression model with adaptive learning rates.
    """

    def __init__(self, *args):
        super().__init__(*args)
        #self.training_data.set_normalization_min_max()
        # Initialize adaptive learning rates for each parameter (weights + bias)
        self.adaptive_lr = [self.learning_rate] * (len(self.weights) + 1)

    def update_learning_rate(self, index, error_magnitude):
        """
        Adjust the learning rate for a specific parameter based on error magnitude.
        """
        decay_factor = 0.001  # Decay factor to gradually reduce the learning rate
        self.adaptive_lr[index] /= (1 + decay_factor * abs(error_magnitude))

    def training_iteration(self, training_data) -> float:
        inp_0 = training_data[0]                     # First input feature
        inp_1 = training_data[1]                     # Second input feature
        target = training_data[-1]                   # Target value

        prediction = inp_0 * self.weights[0] + self.bias + inp_1 * self.weights[1]  # Step 1: Prediction
        error = target - prediction                  # Step 2: Calculate error

        print (f"epoch:{self.metrics_mgr.epoch_curr_number} self.adaptive_lr = {self.adaptive_lr} ")

        # Update each weight and bias with its adaptive learning rate
        self.update_learning_rate(0, error * inp_0)  # Update learning rate for weight[0]
        self.weights[0] += self.adaptive_lr[0] * error * inp_0

        self.update_learning_rate(1, error * inp_1)  # Update learning rate for weight[1]
        self.weights[1] += self.adaptive_lr[1] * error * inp_1

        self.update_learning_rate(2, error)          # Update learning rate for bias
        self.bias += self.adaptive_lr[2] * error

        return prediction
