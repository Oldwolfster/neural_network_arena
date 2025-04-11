from src.engine.BaseGladiator import Gladiator


class HondaBlackBird(Gladiator):
    """
    A simple single input regression model
    This class serves as a template for more complex implementations.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def training_iteration(self, training_data) -> float:
        input            = training_data[0]                     # First Element of each tuple
        target           = training_data[-1]                    # Last element is target

        prediction       = input * self.weights[0] + self.bias  # Step 1) Guess
        error            = target - prediction                  # Step 2) Check guess
        self.weights[0] += error * self.learning_rate * input   # Step 3) Adjust Weight - formula for gradient descent
        self.bias       += error * self.learning_rate           # Step 3) Adjust Bias
        return prediction