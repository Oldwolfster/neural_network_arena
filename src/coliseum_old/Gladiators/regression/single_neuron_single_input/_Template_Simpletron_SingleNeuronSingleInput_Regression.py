from src.NNA.engine.Metrics import GladiatorOutput
from src.NNA.engine.BaseGladiator import Gladiator


class _Template_Simpletron_Regression(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.bias = .53
        self.weights = [0.1]


    def training_iteration(self, training_data) -> GladiatorOutput:
        #print(f"Model:{self.metrics_mgr.name} Epoch:{self.metrics_mgr.epoch_curr_number}\tIteration:{self.metrics_mgr.iteration_num}")

        input = training_data[0]
        target = training_data[-1]

        prediction  = input * self.weights[0] + self.bias           # Step 1) Guess
        error       = target - prediction                       # Step 2) Check guess,
        adjustment  = error * self.learning_rate                #         If wrong, how much

        self.weights[0] += adjustment                  # Step 3) Adjust(Apply)
        self.bias   += adjustment

        return prediction


