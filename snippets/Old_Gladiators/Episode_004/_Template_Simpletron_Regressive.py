from src.Engine import *
from src.Metrics import GladiatorOutput
from src.BaseGladiator import Gladiator

class _Template_Simpletron_Regressive(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.
    it is designed for very simple training data, to be able to easily see
    how and why it performs like it does.
    """

    def __init__(self, number_of_epochs: int, metrics_mgr: MetricsMgr, *args):
        super().__init__(number_of_epochs, metrics_mgr, *args)

    def training_iteration(self, training_data) -> GladiatorOutput:
        credit_score= training_data[0]
        repay_ratio = training_data[1]
        prediction  = self.predict(credit_score)                              # Step 1) Guess
        error       = self.compare(prediction, repay_ratio)                   # Step 2) Check guess, if wrong, by how much and quare it (MSE)
        adjustment  = self.adjust_weight(error)                         # Step 3) Adjust(Calc)                                       # For MSE the loss is the error squared and cut in half
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

    def predict(self, credit_score: float) -> float:
        return credit_score * self.weight                               # Simplifies prediction as no step function



    def compare(self, prediction: float, target: float) -> float:            # Calculate the Loss (Just MSE for now)
        return (target - prediction)

    def adjust_weight(self, loss: float) -> float:                       # Apply learning rate to loss.
        # Chat GPT says it should be 2 * Loss * Learning Rate * input
        return loss * self.learning_rate
