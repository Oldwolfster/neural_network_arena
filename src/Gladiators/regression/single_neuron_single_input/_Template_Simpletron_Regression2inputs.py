from src.engine.Metrics import GladiatorOutput
from src.engine.BaseGladiator import Gladiator


class _Template_Simpletron_Regression2inputs(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.bias = .53
        self.weight = .1


    def training_iteration(self, training_data) -> GladiatorOutput:
        input = training_data[0]
        target = training_data[2]
        prediction  = input * self.weight + self.bias           # Step 1) Guess
        error       = target - prediction                       # Step 2) Check guess,
        adjustment  = error * self.learning_rate                #         if wrong, how much
        new_weight  = self.weight + adjustment                  # Step 3) Adjust(Apply)
        new_bias    = self.bias + adjustment
        gladiator_output    = GladiatorOutput(
            prediction      = prediction,
            adjustment      = adjustment,
            weight          = self.weight,
            new_weight      = new_weight,
            bias            = self.bias,
            new_bias        = new_bias
        )
        self.weight = new_weight
        self.bias   = new_bias
        return gladiator_output

