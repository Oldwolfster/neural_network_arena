from src.Arena import *
from src.Metrics import Metrics,
from src.Gladiator import Gladiator

class _Template_Simpletron_Regressive(Gladiator):
    """
    A simple perceptron implementation for educational purposes.
    This class serves as a template for more complex implementations.
    t is designed for very simple training data, to be able to easily see
    how and why it performs like it does.
     regression training data.
    It calculates a credit score between 0-100 and uses it to generate a continuous target value
    representing the repayment ratio (0.0 to 1.0).(with random noise thrown in)
    """

    def __init__(self, number_of_epochs: int, metrics: Metrics, *args):
        super().__init__(number_of_epochs, metrics, *args)

    def training_iteration(self, training_data) -> IterationData:
        crdit_score = training_data[0]
        repay_ratio = training_data[1]
        prediction  = self.predict(crdit_score)                              # Step 1) Guess
        error       = self.compare(prediction, repay_ratio)                   # Step 2) Check guess, if wrong, by how much and quare it (MSE)
        adjustment  = self.adjust_weight(error)                         # Step 3) Adjust(Calc)
        loss        = (error ** 2) * 0.5                                # For MSE the loss is the error squared and cut in half
        new_weight  = self.weight + adjustment                          # Step 3) Adjust(Apply)
        data = IterationData(
            prediction=prediction,
            adjustment=adjustment,
            weight=self.weight,
            new_weight=new_weight,
            bias=0.0,
            new_bias=0.0
        )
        self.weight = new_weight
        return data

    def predict(self, credit_score: float) -> float:
        return credit_score * self.weight                               # Simplifies prediction as no step function



    def compare(self, prediction: float, target: float) -> float:            # Calculate the Loss (Just MSE for now)
        return (target - prediction)

    def adjust_weight(self, loss: float) -> float:                       # Apply learning rate to loss.
        # Chat GPT says it should be 2 * Loss * Learning Rate * input
        return loss * self.learning_rate
