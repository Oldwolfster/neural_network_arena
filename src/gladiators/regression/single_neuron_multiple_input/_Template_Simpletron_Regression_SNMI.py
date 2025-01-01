from src.engine.Metrics import GladiatorOutput
from src.engine.BaseGladiator import Gladiator


class _Template_Simpletron_Regression(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.bias = .53
        self.weight = .1


    def training_iteration(self, training_data) -> GladiatorOutput:
        #print(f"Model:{self.metrics_mgr.name} Epoch:{self.metrics_mgr.epoch_curr_number}\tIteration:{self.metrics_mgr.iteration_num}")

        input = training_data[0]
        target = training_data[-1]

        prediction  = input * self.weight + self.bias           # Step 1) Guess
        error       = target - prediction                       # Step 2) Check guess,
        adjustment  = error * self.learning_rate                #         If wrong, how much
        new_weight  = self.weight + adjustment                  # Step 3) Adjust(Apply)
        new_bias    = self.bias + adjustment
        gladiator_output    = GladiatorOutput(
            prediction      = prediction
        )
        self.weight = new_weight
        self.bias   = new_bias

        return gladiator_output


