from src.engine.Metrics import GladiatorOutput
from src.gladiators.BaseGladiator import Gladiator


class _Template_Simpletron(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)

    def training_iteration(self, training_data) -> GladiatorOutput:
        input = training_data[0]
        target = training_data[1]
        prediction  = input * self.weight                       # Step 1) Guess
        prediction = 1 if round(prediction,7) >= 0.5 else 0     #         Step Function to make it 0 or 1
        error        = target - prediction                      # Step 2) Check guess,
        adjustment  = error * self.learning_rate                #         if wrong, how much
        new_weight  = self.weight + adjustment                  # Step 3) Adjust(Apply)
        gladiator_output = GladiatorOutput(
            prediction=prediction,
            adjustment=adjustment,
            weight=self.weight,
            new_weight=new_weight,
            bias=0.0,
            new_bias=0.0
        )
        self.weight = new_weight
        return gladiator_output


