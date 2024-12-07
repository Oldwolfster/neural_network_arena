from src.engine.Metrics import GladiatorOutput
from src.gladiators.BaseGladiator import Gladiator


class Blackbird(Gladiator):
    """
    A simple perceptron using Regression
    This class serves as a template for more complex implementations.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)

    def training_iteration(self, training_data) -> GladiatorOutput:
        input = training_data[0]
        result = training_data[1]
        prediction  = self.predict(input)                               # Step 1) Guess
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
        return credit_score * self.weight                               # Rounded to 7 decimals to avoid FP errors

    def compare(self, prediction: float, result: float) -> float:            # Calculate the Loss
        return result - prediction

    def adjust_weight(self, loss: float) -> float:                       # Apply learning rate to loss.
        return loss * self.learning_rate
