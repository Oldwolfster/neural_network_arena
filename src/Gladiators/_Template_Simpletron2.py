from src.ArenaEngine import *
from src.Metrics import GladiatorOutput
from src.BaseGladiator import Gladiator


class _Template_Simpletron2(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)

    def training_iteration(self, training_data) -> GladiatorOutput:
        inpt = training_data[0]
        target = training_data[1]
        prediction  = input * self.weight # Step 1) Guess
        prediction = 1 if round(prediction,7) >= 0.5 else 0
        loss        = target - prediction # Step 2) Check guess, if wrong, how much?
        adjustment  = loss * self.learning_rate # Step 3) Adjust(Calc)
        new_weight  = self.weight + adjustment                          # Step 3) Adjust(Apply)
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

    def predict(self, credit_score: float) -> int:
        return 1 if round(credit_score * self.weight, 7) >= 0.5 else 0   # Rounded to 7 decimals to avoid FP errors

    def compare(self, prediction: float, result: float) -> float:            # Calculate the Loss
        return result - prediction

    def adjust_weight(self, loss: float) -> float:                       # Apply learning rate to loss.
        return loss * self.learning_rate
