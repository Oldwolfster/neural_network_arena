from src.Arena import *
from src.Metrics import GladiatorOutput
from src.Gladiator import Gladiator


class _Template_Simpletron2(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.

    The training data is a tuple where the first element is the input and the second is the target
    After processing return a GladiatorOutput objeect with the fields populated.  Neural Network Arena will handle the rest
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)

    def training_iteration(self, training_data) -> GladiatorOutput:
        credit_score = training_data[0]
        result = training_data[1]
        prediction  = self.predict(credit_score)                        # Step 1) Guess
        loss        = self.compare(prediction, result)                  # Step 2) Check guess, if wrong, how much?
        adjustment  = self.adjust_weight(loss)                          # Step 3) Adjust(Calc)
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
